"""MS2 spectral-library retrieval and candidate ranking.

For each molecule, compare every query spectrum with reference spectra belonging
to its MS1-retrieved candidate structures. A candidate's score is the maximum
matchms cosine similarity over all query/reference spectrum pairs.

The reference library is streamed from Parquet in batches and only spectra whose
inchikey14 occurs in the MS1 candidate set are retained. This avoids loading the
full 2.5M-spectrum peak library into memory.

Run from the repository root:
    python runms2.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matchms import Spectrum
from matchms.similarity import CosineGreedy


PEAK_MZ_COLUMN = "ms2_mzs"
PEAK_INTENSITY_COLUMN = "ms2_normalized_intensities"
KEY_COLUMN = "inchikey14"


def _as_peak_arrays(mzs: Any, intensities: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, positive-intensity peaks sorted by m/z."""
    if mzs is None or intensities is None:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    try:
        mz = np.asarray(mzs, dtype=np.float32).reshape(-1)
        intensity = np.asarray(intensities, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    if mz.size != intensity.size or mz.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[valid], intensity[valid]
    if mz.size == 0:
        return mz, intensity

    order = np.argsort(mz, kind="stable")
    return mz[order], intensity[order]


def make_spectrum(mzs: Any, intensities: Any) -> Spectrum | None:
    """Create a matchms Spectrum, returning None for unusable peak lists."""
    mz, intensity = _as_peak_arrays(mzs, intensities)
    if mz.size < 2:
        return None

    # Matchms performs intensity normalization as part of its spectrum handling;
    # explicit peak sorting/filtering here keeps malformed rows from poisoning
    # an entire run.
    return Spectrum(mz=mz, intensities=intensity, metadata={})


def _required_columns(path: Path, required: set[str]) -> list[str]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = required - available
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Available columns include: {sorted(available)[:50]}"
        )
    return sorted(required)


def load_test_spectra(test_path: Path) -> pd.DataFrame:
    required = {
        "molecule_id",
        "spectrum_id",
        PEAK_MZ_COLUMN,
        PEAK_INTENSITY_COLUMN,
    }
    columns = _required_columns(test_path, required)
    df = pd.read_parquet(test_path, columns=columns)
    df = df.dropna(subset=["molecule_id", "spectrum_id"]).copy()
    print(f"Loaded query spectra: {len(df):,}")
    print(f"Query molecules:       {df['molecule_id'].nunique():,}")
    return df


def load_ms1_candidates(candidate_path: Path) -> pd.DataFrame:
    required = {
        "molecule_id",
        "spectrum_id",
        "candidate_id",
        "normalized_smiles",
        "inchikey14",
        "mass_error_ppm",
    }
    df = pd.read_parquet(candidate_path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{candidate_path} is missing required columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["molecule_id", KEY_COLUMN, "normalized_smiles"]).copy()
    df[KEY_COLUMN] = df[KEY_COLUMN].astype(str)
    df = df[df[KEY_COLUMN].str.len() > 0]
    if df.empty:
        raise ValueError("MS1 candidate file contains no usable candidates.")

    # A molecule can have multiple spectra, and each spectrum can have the same
    # candidate. Keep the best absolute precursor mass error for tie-breaking.
    df["abs_mass_error_ppm"] = pd.to_numeric(
        df["mass_error_ppm"], errors="coerce"
    ).abs()
    print(f"MS1 candidate rows:    {len(df):,}")
    print(f"Candidate molecules:   {df['molecule_id'].nunique():,}")
    print(f"Unique candidate keys: {df[KEY_COLUMN].nunique():,}")
    return df


def load_candidate_reference_spectra(
    train_path: Path,
    candidate_keys: set[str],
    batch_size: int = 65_536,
) -> dict[str, list[Spectrum]]:
    """Stream training Parquet, retaining spectra for relevant candidate keys."""
    required = {
        KEY_COLUMN,
        PEAK_MZ_COLUMN,
        PEAK_INTENSITY_COLUMN,
    }
    columns = _required_columns(train_path, required)
    parquet = pq.ParquetFile(train_path)
    library: dict[str, list[Spectrum]] = defaultdict(list)

    total_rows = 0
    retained_rows = 0
    print("\nScanning training spectra in batches...")
    for batch_number, batch in enumerate(
        parquet.iter_batches(batch_size=batch_size, columns=columns), start=1
    ):
        frame = batch.to_pandas()
        total_rows += len(frame)
        frame = frame[frame[KEY_COLUMN].astype("string").isin(candidate_keys)]
        retained_rows += len(frame)

        for key, mzs, intensities in zip(
            frame[KEY_COLUMN].tolist(),
            frame[PEAK_MZ_COLUMN].tolist(),
            frame[PEAK_INTENSITY_COLUMN].tolist(),
        ):
            if key is None or pd.isna(key):
                continue
            spectrum = make_spectrum(mzs, intensities)
            if spectrum is not None:
                library[str(key)].append(spectrum)

        if batch_number % 10 == 0:
            print(
                f"  Scanned {total_rows:,} rows; "
                f"retained {retained_rows:,} candidate-key rows"
            )

    print(f"Training rows scanned:       {total_rows:,}")
    print(f"Candidate-key rows retained: {retained_rows:,}")
    print(f"Keys with usable references: {len(library):,}")
    print(
        "Reference spectra retained: "
        f"{sum(len(spectra) for spectra in library.values()):,}"
    )
    return dict(library)


def rank_candidates(
    test_spectra: pd.DataFrame,
    candidates: pd.DataFrame,
    reference_library: dict[str, list[Spectrum]],
    tolerance_da: float = 0.01,
) -> pd.DataFrame:
    """Score candidates molecule-wise and return every candidate ranked."""
    cosine = CosineGreedy(
        tolerance=float(tolerance_da),
        mz_power=0,
        intensity_power=0.5,
    )

    # Build query Spectrum objects once, keyed by spectrum_id.
    query_spectra: dict[Any, Spectrum | None] = {}
    for spectrum_id, mzs, intensities in zip(
        test_spectra["spectrum_id"].tolist(),
        test_spectra[PEAK_MZ_COLUMN].tolist(),
        test_spectra[PEAK_INTENSITY_COLUMN].tolist(),
    ):
        query_spectra[spectrum_id] = make_spectrum(mzs, intensities)

    # Cache query/key scores because the same candidate may be proposed by
    # several query spectra, and the same spectrum can be encountered again.
    pair_cache: dict[tuple[Any, str], float] = {}
    results: list[dict[str, Any]] = []
    molecule_groups = candidates.groupby("molecule_id", sort=False)

    print("\nScoring molecule-level candidate sets...")
    for molecule_number, (molecule_id, molecule_candidates) in enumerate(
        molecule_groups, start=1
    ):
        query_ids = (
            test_spectra.loc[
                test_spectra["molecule_id"] == molecule_id, "spectrum_id"
            ]
            .drop_duplicates()
            .tolist()
        )
        if not query_ids:
            continue

        # One output row per candidate structure key per molecule.
        candidate_rows = (
            molecule_candidates.sort_values(
                ["abs_mass_error_ppm", "candidate_id"],
                na_position="last",
                kind="mergesort",
            )
            .drop_duplicates(KEY_COLUMN, keep="first")
        )

        for _, candidate in candidate_rows.iterrows():
            key = str(candidate[KEY_COLUMN])
            references = reference_library.get(key, [])
            best_score = 0.0
            scored_pairs = 0

            if references:
                for spectrum_id in query_ids:
                    query = query_spectra.get(spectrum_id)
                    if query is None:
                        continue

                    cache_key = (spectrum_id, key)
                    if cache_key not in pair_cache:
                        maximum = 0.0
                        for reference in references:
                            try:
                                value = cosine(query, reference)
                                score = float(value[0] if isinstance(value, (tuple, list, np.ndarray)) else value)
                                if np.isfinite(score) and score > maximum:
                                    maximum = score
                            except (ValueError, TypeError, IndexError, FloatingPointError):
                                continue
                        pair_cache[cache_key] = maximum

                    best_score = max(best_score, pair_cache[cache_key])
                    scored_pairs += 1

            results.append(
                {
                    "molecule_id": molecule_id,
                    "candidate_id": candidate["candidate_id"],
                    "normalized_smiles": candidate["normalized_smiles"],
                    "inchikey14": key,
                    "ms2_max_cosine": float(best_score),
                    "reference_spectrum_count": len(references),
                    "query_reference_pairs_scored": scored_pairs,
                    "best_abs_mass_error_ppm": candidate["abs_mass_error_ppm"],
                }
            )

        if molecule_number % 50 == 0:
            print(
                f"  Molecules: {molecule_number:,}/{len(molecule_groups):,}; "
                f"cached query/candidate scores: {len(pair_cache):,}"
            )

    ranked = pd.DataFrame(results)
    if ranked.empty:
        raise RuntimeError("MS2 scoring produced no candidate rows.")

    # Spectral score is primary. Mass error is a deterministic tie-breaker;
    # candidate_id makes output stable across runs.
    ranked = ranked.sort_values(
        [
            "molecule_id",
            "ms2_max_cosine",
            "best_abs_mass_error_ppm",
            "candidate_id",
        ],
        ascending=[True, False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = ranked.groupby("molecule_id", sort=False).cumcount() + 1
    ranked["is_top25"] = ranked["rank"] <= 25
    return ranked


def print_summary(ranked: pd.DataFrame, library: dict[str, list[Spectrum]]) -> None:
    molecules = ranked["molecule_id"].nunique()
    top25 = ranked[ranked["rank"] <= 25]
    print("\n" + "=" * 68)
    print("MS2 SPECTRAL RETRIEVAL SUMMARY")
    print("=" * 68)
    print(f"Molecules ranked:               {molecules:,}")
    print(f"Candidate structures ranked:    {len(ranked):,}")
    print(f"Candidates with reference MS2:   {(ranked['reference_spectrum_count'] > 0).sum():,}")
    print(f"Candidates without reference:    {(ranked['reference_spectrum_count'] == 0).sum():,}")
    print(f"Mean best cosine:                {ranked['ms2_max_cosine'].mean():.5f}")
    print(f"Median best cosine:              {ranked['ms2_max_cosine'].median():.5f}")
    print(f"Keys with reference spectra:     {len(library):,}")
    print(f"Rows in top 25:                  {len(top25):,}")
    print("\nTop 10 example rows:")
    print(
        ranked[
            [
                "molecule_id",
                "rank",
                "inchikey14",
                "ms2_max_cosine",
                "reference_spectrum_count",
                "best_abs_mass_error_ppm",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


__all__ = [
    "load_test_spectra",
    "load_ms1_candidates",
    "load_candidate_reference_spectra",
    "rank_candidates",
    "print_summary",
]
