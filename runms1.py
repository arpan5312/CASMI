from pathlib import Path

import numpy as np
import pandas as pd

from candidate_generation.ms1 import append_neutral_mass


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CASMI = ROOT / "CASMI"

TEST_PATH = CASMI / "data" / "test.parquet"
CANDIDATE_DB_PATH = CASMI / "data" / "candidate_db.csv"
OUTPUT_PATH = CASMI / "data" / "ms1_candidates.parquet"

TOLERANCE_PPM = 5.0


# ============================================================
# LOAD DATA
# ============================================================

def load_test_data():
    print(f"Loading test data: {TEST_PATH}")

    df = pd.read_parquet(TEST_PATH)

    required = {
        "molecule_id",
        "spectrum_id",
        "adduct",
        "precursor_mz",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Test data is missing required columns: {sorted(missing)}"
        )

    print(f"Test spectra: {len(df):,}")

    return df


def load_candidate_db():
    print(f"Loading candidate DB: {CANDIDATE_DB_PATH}")

    df = pd.read_csv(CANDIDATE_DB_PATH)

    required = {
        "candidate_id",
        "normalized_smiles",
        "inchikey",
        "inchikey14",
        "molecular_formula",
        "neutral_mass",
        "source",
        "in_train",
        "in_coconut",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Candidate DB is missing required columns: {sorted(missing)}"
        )

    df["neutral_mass"] = pd.to_numeric(
        df["neutral_mass"],
        errors="coerce",
    )

    if df["neutral_mass"].isna().any():
        bad = int(df["neutral_mass"].isna().sum())
        raise ValueError(
            f"Candidate DB contains {bad:,} invalid neutral masses."
        )

    # CRITICAL:
    # MS1 retrieval needs candidates sorted by neutral mass.
    # DO NOT set candidate_id as the dataframe index.
    df = (
        df.sort_values(
            "neutral_mass",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    print(f"Candidate structures: {len(df):,}")

    return df


# ============================================================
# MS1 RETRIEVAL
# ============================================================

def retrieve_candidates(test_df, candidate_db):
    """
    MS1 candidate retrieval.

    For every test spectrum:

        precursor m/z
              ↓
        experimental neutral mass
              ↓
        ± tolerance ppm mass window
              ↓
        binary search candidate DB
              ↓
        candidate structures

    Candidate DB is sorted by neutral_mass.
    """

    print(
        f"\nRunning MS1 retrieval at ±{TOLERANCE_PPM} ppm..."
    )

    # This is the sorted mass index.
    candidate_masses = candidate_db["neutral_mass"].to_numpy(
        dtype=np.float64
    )

    results = []

    for row_number, (_, spectrum) in enumerate(
        test_df.iterrows()
    ):

        query_mass = spectrum["neutral_mass"]

        if pd.isna(query_mass):
            continue

        query_mass = float(query_mass)

        # ppm → absolute Da tolerance
        tolerance_da = (
            query_mass * TOLERANCE_PPM / 1_000_000.0
        )

        lower_mass = query_mass - tolerance_da
        upper_mass = query_mass + tolerance_da

        # Binary search into the sorted candidate mass array.
        left = np.searchsorted(
            candidate_masses,
            lower_mass,
            side="left",
        )

        right = np.searchsorted(
            candidate_masses,
            upper_mass,
            side="right",
        )

        molecule_id = spectrum["molecule_id"]
        spectrum_id = spectrum["spectrum_id"]

        # Every candidate inside the MS1 mass window.
        for candidate_idx in range(left, right):

            candidate = candidate_db.iloc[candidate_idx]

            candidate_mass = float(
                candidate["neutral_mass"]
            )

            ppm_error = (
                (candidate_mass - query_mass)
                / query_mass
                * 1_000_000.0
            )

            results.append(
                {
                    "molecule_id": molecule_id,
                    "spectrum_id": spectrum_id,

                    "candidate_id": candidate["candidate_id"],
                    "normalized_smiles": candidate[
                        "normalized_smiles"
                    ],
                    "inchikey": candidate["inchikey"],
                    "inchikey14": candidate["inchikey14"],
                    "molecular_formula": candidate[
                        "molecular_formula"
                    ],

                    "experimental_neutral_mass": query_mass,
                    "candidate_neutral_mass": candidate_mass,
                    "mass_error_ppm": ppm_error,

                    "source": candidate["source"],
                    "in_train": candidate["in_train"],
                    "in_coconut": candidate["in_coconut"],
                }
            )

    result_df = pd.DataFrame(results)

    return result_df


# ============================================================
# SUMMARY
# ============================================================

def print_summary(test_df, candidates):
    print("\n" + "=" * 60)
    print("MS1 RETRIEVAL SUMMARY")
    print("=" * 60)

    if candidates.empty:
        print("NO CANDIDATES FOUND.")
        return

    spectra_with_candidates = (
        candidates["spectrum_id"]
        .nunique()
    )

    total_spectra = len(test_df)

    candidate_counts = (
        candidates
        .groupby("spectrum_id")
        .size()
    )

    print(
        f"Spectra:              {total_spectra:,}"
    )

    print(
        f"Spectra w/ candidates: {spectra_with_candidates:,}"
    )

    print(
        f"Total candidate rows:  {len(candidates):,}"
    )

    print(
        f"Mean candidates:       {candidate_counts.mean():,.2f}"
    )

    print(
        f"Median candidates:     {candidate_counts.median():,.0f}"
    )

    print(
        f"Min candidates:        {candidate_counts.min():,}"
    )

    print(
        f"Max candidates:        {candidate_counts.max():,}"
    )

    print(
        f"Mean |ppm error|:      "
        f"{candidates['mass_error_ppm'].abs().mean():.3f}"
    )

    print(
        f"Median |ppm error|:    "
        f"{candidates['mass_error_ppm'].abs().median():.3f}"
    )

    print("\nCandidate source breakdown:")

    source_counts = (
        candidates["source"]
        .value_counts()
    )

    for source, count in source_counts.items():
        print(
            f"  {source:<20} {count:,}"
        )

    print("\nMass-error percentiles:")

    percentiles = (
        candidates["mass_error_ppm"]
        .abs()
        .quantile(
            [0.50, 0.75, 0.90, 0.95, 0.99]
        )
    )

    for percentile, value in percentiles.items():
        print(
            f"  {percentile * 100:>5.1f}% : {value:.3f} ppm"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CASMI MS1 CANDIDATE RETRIEVAL")
    print("=" * 60)

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    test_df = load_test_data()

    print("\nCalculating experimental neutral masses...")

    test_df = append_neutral_mass(test_df)

    valid_masses = (
        test_df["neutral_mass"]
        .notna()
        .sum()
    )

    print(
        f"Valid neutral masses: "
        f"{valid_masses:,}/{len(test_df):,}"
    )

    # --------------------------------------------------------
    # Candidate DB
    # --------------------------------------------------------

    candidate_db = load_candidate_db()

    # --------------------------------------------------------
    # MS1 retrieval
    # --------------------------------------------------------

    candidates = retrieve_candidates(
        test_df,
        candidate_db,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    candidates.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved candidates to:"
        f"\n{OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        test_df,
        candidates,
    )

    print("\n" + "=" * 60)
    print("MS1 PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()