"""MS1-only candidate retrieval baseline for CASMI.

Evaluates molecule-level retrieval from precursor m/z + adduct only.

The validation split is made by molecule_id. Validation structures are removed
from the candidate DB by inchikey14 so the experiment measures retrieval for
structures unseen by the candidate library.

Usage:
    python -m src.evaluation.ms1_baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.candidate_generation.ms1 import append_neutral_mass, find_candidates


DEFAULT_TOLERANCES = (5.0, 10.0, 20.0, 30.0)


def load_train(path: Path) -> pd.DataFrame:
    cols = [
        "molecule_id",
        "spectrum_id",
        "adduct",
        "precursor_mz",
        "inchikey14",
    ]
    df = pd.read_parquet(path, columns=cols)
    df = df.dropna(subset=["molecule_id", "adduct", "precursor_mz", "inchikey14"])
    df["precursor_mz"] = pd.to_numeric(df["precursor_mz"], errors="coerce")
    return df.dropna(subset=["precursor_mz"])


def make_molecule_split(
    train: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by molecule_id so spectra from one molecule never cross the split."""
    molecule_ids = train["molecule_id"].drop_duplicates().to_numpy()

    rng = np.random.default_rng(seed)
    rng.shuffle(molecule_ids)

    n_val = max(1, int(len(molecule_ids) * validation_fraction))
    val_ids = set(molecule_ids[:n_val])

    val = train[train["molecule_id"].isin(val_ids)].copy()
    return train[~train["molecule_id"].isin(val_ids)].copy(), val


def prepare_candidate_db(
    candidate_db_path: Path,
    heldout_inchikey14: set[str],
) -> pd.DataFrame:
    cols = [
        "candidate_id",
        "normalized_smiles",
        "inchikey14",
        "neutral_mass",
        "source",
    ]
    db = pd.read_csv(candidate_db_path, usecols=cols)

    # Prevent the validation answer from leaking through COCONUT/train overlap.
    db = db[~db["inchikey14"].isin(heldout_inchikey14)].copy()
    db["neutral_mass"] = pd.to_numeric(db["neutral_mass"], errors="coerce")
    db = db.dropna(subset=["neutral_mass"])
    db = db.sort_values("neutral_mass", kind="mergesort").reset_index(drop=True)
    return db


def molecule_retrieval(
    spectra: pd.DataFrame,
    candidate_db: pd.DataFrame,
    tolerance_ppm: float,
) -> dict:
    """Retrieve candidates for every spectrum, then aggregate at molecule level.

    A candidate's molecule-level score is its best (minimum) absolute ppm
    precursor-mass error across all spectra belonging to that molecule.
    """
    records = []

    for molecule_id, group in spectra.groupby("molecule_id", sort=False):
        true_keys = group["inchikey14"].dropna().unique()
        if len(true_keys) != 1:
            continue
        true_key = true_keys[0]

        best_error: dict[str, float] = {}
        best_candidate: dict[str, int] = {}

        for _, row in group.iterrows():
            query = pd.DataFrame(
                {
                    "adduct": [row["adduct"]],
                    "precursor_mz": [row["precursor_mz"]],
                }
            )
            try:
                query = append_neutral_mass(query)
                neutral_mass = float(query.iloc[0]["neutral_mass"])
            except (TypeError, ValueError, KeyError):
                continue

            candidates = find_candidates(
                candidate_db.set_index("neutral_mass", drop=False),
                neutral_mass,
                tolerance_ppm=tolerance_ppm,
            )

            if candidates.empty:
                continue

            errors = (
                (candidates["neutral_mass"].to_numpy() - neutral_mass)
                / neutral_mass
                * 1_000_000
            )

            for candidate_id, ppm_error in zip(
                candidates["candidate_id"].to_numpy(),
                np.abs(errors),
            ):
                candidate_id = int(candidate_id)
                if candidate_id not in best_error or ppm_error < best_error[candidate_id]:
                    best_error[candidate_id] = float(ppm_error)
                    best_candidate[candidate_id] = candidate_id

        if not best_error:
            records.append(
                {
                    "molecule_id": molecule_id,
                    "true_inchikey14": true_key,
                    "candidate_count": 0,
                    "rank": np.nan,
                    "reciprocal_rank": 0.0,
                    "hit_at_1": 0,
                    "hit_at_10": 0,
                    "hit_at_25": 0,
                    "hit_at_100": 0,
                }
            )
            continue

        ranked = sorted(best_error.items(), key=lambda x: (x[1], x[0]))
        ranked_ids = [candidate_id for candidate_id, _ in ranked]

        true_rows = candidate_db[
            candidate_db["candidate_id"].isin(ranked_ids)
            & (candidate_db["inchikey14"] == true_key)
        ]

        true_candidate_ids = set(true_rows["candidate_id"].astype(int))
        rank = next(
            (i + 1 for i, candidate_id in enumerate(ranked_ids) if candidate_id in true_candidate_ids),
            None,
        )

        records.append(
            {
                "molecule_id": molecule_id,
                "true_inchikey14": true_key,
                "candidate_count": len(ranked_ids),
                "rank": rank,
                "reciprocal_rank": 1.0 / rank if rank is not None else 0.0,
                "hit_at_1": int(rank is not None and rank <= 1),
                "hit_at_10": int(rank is not None and rank <= 10),
                "hit_at_25": int(rank is not None and rank <= 25),
                "hit_at_100": int(rank is not None and rank <= 100),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.parquet"))
    parser.add_argument(
        "--candidate-db",
        type=Path,
        default=Path("data/candidate_db.csv"),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES),
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CASMI MS1-ONLY RETRIEVAL BASELINE")
    print("=" * 70)

    print("\n[1/4] Loading train spectra...")
    train = load_train(args.train)
    print(f"  Rows: {len(train):,}")
    print(f"  Molecules: {train['molecule_id'].nunique():,}")

    print("\n[2/4] Creating molecule-level validation split...")
    _, validation = make_molecule_split(
        train,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    heldout_keys = set(validation["inchikey14"].astype(str))
    print(f"  Validation molecules: {validation['molecule_id'].nunique():,}")
    print(f"  Validation spectra:   {len(validation):,}")
    print(f"  Held-out structures:  {len(heldout_keys):,}")

    print("\n[3/4] Loading candidate database...")
    candidate_db = prepare_candidate_db(args.candidate_db, heldout_keys)
    print(f"  Candidates after leakage removal: {len(candidate_db):,}")

    # Building this once avoids repeatedly constructing the same index.
    candidate_db = candidate_db.set_index("neutral_mass", drop=False)

    print("\n[4/4] Evaluating...")
    rows = []
    for tolerance in args.tolerances:
        result = molecule_retrieval(
            validation,
            candidate_db,
            tolerance_ppm=tolerance,
        )

        evaluated = len(result)
        rows.append(
            {
                "tolerance_ppm": tolerance,
                "molecules": evaluated,
                "mean_candidates": result["candidate_count"].mean(),
                "median_candidates": result["candidate_count"].median(),
                "hit@1": result["hit_at_1"].mean(),
                "hit@10": result["hit_at_10"].mean(),
                "hit@25": result["hit_at_25"].mean(),
                "hit@100": result["hit_at_100"].mean(),
                "MRR": result["reciprocal_rank"].mean(),
            }
        )

    summary = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("MS1 BASELINE RESULTS")
    print("=" * 70)
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_candidates": "{:.2f}".format,
                "median_candidates": "{:.0f}".format,
                "hit@1": "{:.4f}".format,
                "hit@10": "{:.4f}".format,
                "hit@25": "{:.4f}".format,
                "hit@100": "{:.4f}".format,
                "MRR": "{:.4f}".format,
            },
        )
    )

    output = Path("data/ms1_baseline_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
