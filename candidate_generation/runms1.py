from pathlib import Path

import pandas as pd

from candidate_generation.ms1 import append_neutral_mass, find_candidates

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT / "data" / "test.parquet"
CANDIDATE_DB_PATH = ROOT / "data" / "candidate_db.csv"
OUTPUT_PATH = ROOT / "data" / "ms1_candidates.parquet"

TOLERANCE_PPM = 5.0


# ============================================================
# LOAD DATA
# ============================================================

def load_test_data():
    print(f"Loading test data: {TEST_PATH}")
    df = pd.read_parquet(TEST_PATH)

    required = {"adduct", "precursor_mz"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Test data is missing required columns: {sorted(missing)}"
        )

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
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Candidate DB is missing required columns: {sorted(missing)}"
        )

    # The binary-search implementation requires the mass index
    # to be sorted numerically.
    df = (
        df.sort_values("neutral_mass")
        .reset_index(drop=True)
    )

    df = df.set_index("neutral_mass", drop=False)

    return df


# ============================================================
# MS1 RETRIEVAL
# ============================================================

def retrieve_candidates(test_df, candidate_db):
    results = []

    print(
        f"\nRunning MS1 retrieval at ±{TOLERANCE_PPM} ppm..."
    )

    for row_number, (_, spectrum) in enumerate(test_df.iterrows()):

        query_mass = spectrum["neutral_mass"]

        # Invalid neutral mass means this spectrum cannot
        # participate in MS1 retrieval.
        if pd.isna(query_mass):
            continue

        candidates = find_candidates(
            candidate_db,
            query_mass=query_mass,
            tolerance_ppm=TOLERANCE_PPM,
        )

        # ----------------------------------------------------
        # Identify the spectrum.
        #
        # Test data normally has molecule_id and spectrum_id.
        # If either is absent, use the parquet row number.
        # ----------------------------------------------------

        molecule_id = spectrum.get("molecule_id", row_number)
        spectrum_id = spectrum.get("spectrum_id", row_number)

        for candidate_idx in candidates:

            candidate = candidate_db.iloc[candidate_idx]

            candidate_mass = candidate["neutral_mass"]

            error_ppm = (
                (candidate_mass - query_mass)
                / query_mass
                * 1_000_000
            )

            results.append(
                {
                    "molecule_id": molecule_id,
                    "spectrum_id": spectrum_id,

                    "candidate_id": candidate["candidate_id"],

                    "normalized_smiles":
                        candidate["normalized_smiles"],

                    "inchikey":
                        candidate["inchikey"],

                    "inchikey14":
                        candidate["inchikey14"],

                    "molecular_formula":
                        candidate["molecular_formula"],

                    "candidate_neutral_mass":
                        candidate_mass,

                    "experimental_neutral_mass":
                        query_mass,

                    "mass_error_ppm":
                        error_ppm,

                    "source":
                        candidate["source"],

                    "in_train":
                        candidate["in_train"],

                    "in_coconut":
                        candidate["in_coconut"],
                }
            )

        # Progress
        if (row_number + 1) % 100 == 0:
            print(
                f"Processed {row_number + 1}/{len(test_df)} spectra"
            )

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CASMI MS1 CANDIDATE RETRIEVAL")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load test spectra
    # --------------------------------------------------------

    test_df = load_test_data()

    print(f"Test spectra: {len(test_df):,}")

    # --------------------------------------------------------
    # 2. Calculate experimental neutral masses
    # --------------------------------------------------------

    print("\nCalculating experimental neutral masses...")

    test_df = append_neutral_mass(test_df)

    valid_mass = test_df["neutral_mass"].notna().sum()

    print(
        f"Valid neutral masses: "
        f"{valid_mass:,}/{len(test_df):,}"
    )

    # --------------------------------------------------------
    # 3. Load candidate universe
    # --------------------------------------------------------

    candidate_db = load_candidate_db()

    print(
        f"Candidate structures: "
        f"{len(candidate_db):,}"
    )

    # --------------------------------------------------------
    # 4. MS1 retrieval
    # --------------------------------------------------------

    candidates = retrieve_candidates(
        test_df,
        candidate_db,
    )

    # --------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------

    candidates.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Candidate pairs: {len(candidates):,}"
    )

    if len(candidates) > 0:

        counts = (
            candidates
            .groupby("spectrum_id")
            .size()
        )

        print(
            f"\nCandidates per spectrum:"
        )

        print(
            f"  Mean   : {counts.mean():.2f}"
        )
        print(
            f"  Median : {counts.median():.2f}"
        )
        print(
            f"  Min    : {counts.min():,}"
        )
        print(
            f"  Max    : {counts.max():,}"
        )

        print(
            f"\nMass error:"
        )

        print(
            f"  Mean absolute: "
            f"{candidates['mass_error_ppm'].abs().mean():.3f} ppm"
        )

        print(
            f"  Max absolute : "
            f"{candidates['mass_error_ppm'].abs().max():.3f} ppm"
        )

    else:
        print(
            "\nWARNING: No candidates were retrieved."
        )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()