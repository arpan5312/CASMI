"""
Build a unified candidate structure database from:

    1. train.parquet
    2. COCONUT coconutdb.csv

Output:
    data/candidate_db.csv

The database contains one row per unique chemical structure
(deduplicated by inchikey14).

Mass handling:
    - train: calculated from molecular_formula using monoisotopic masses
    - COCONUT: exact_molecular_weight

The resulting database is suitable for the MS1 candidate-retrieval
pipeline.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # repository root (builder lives in data/)

TRAIN_PATH = ROOT / "data" / "train.parquet"
COCONUT_PATH = ROOT / "data" / "coconutdb.csv"
OUTPUT_PATH = ROOT / "data" / "candidate_db.csv"


# ---------------------------------------------------------------------------
# Monoisotopic elemental masses
#
# Kept identical to src/candidate_generation/ms1.py
# ---------------------------------------------------------------------------

ELEMENT_MASSES = {
    "H": 1.00782503223,
    "C": 12.00000000000,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "P": 30.97376199842,
    "S": 31.97207117440,

    "Li": 7.0160034366,
    "Na": 22.9897692820,
    "K": 38.9637064864,
    "Ca": 39.9625908630,

    "Cl": 34.9688526820,
    "Br": 78.9183376,

    "I": 126.904468,
    "Si": 27.97692653465,
    "Se": 79.9165218,
    "As": 74.92159457,
    "B": 11.00930536,
    "Sn": 119.9022026,
}


# ---------------------------------------------------------------------------
# Formula parser
# ---------------------------------------------------------------------------

FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_to_mass(formula):
    """
    Convert a neutral molecular formula into a monoisotopic neutral mass.

    Examples:
        C6H12O6 -> 180.063388...
        C7H8S   -> 124.034670...

    Returns np.nan for invalid/missing formulas.
    """

    if not isinstance(formula, str):
        return np.nan

    formula = formula.strip()

    if not formula:
        return np.nan

    matches = list(FORMULA_PATTERN.finditer(formula))

    # Make sure the entire formula was parsed.
    reconstructed = "".join(match.group(0) for match in matches)

    if reconstructed != formula:
        return np.nan

    mass = 0.0

    for match in matches:
        element = match.group(1)
        count_text = match.group(2)

        if element not in ELEMENT_MASSES:
            return np.nan

        count = int(count_text) if count_text else 1

        mass += ELEMENT_MASSES[element] * count

    return mass


# ---------------------------------------------------------------------------
# Load train structures
# ---------------------------------------------------------------------------

def load_train():
    print("\n[1/5] Loading train structures...")

    # Only load columns actually required.
    columns = [
        "normalized_smiles",
        "inchikey",
        "inchikey14",
        "molecular_formula",
    ]

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=columns,
    )

    print(f"  Train rows: {len(train):,}")

    train = train.rename(
        columns={
            "normalized_smiles": "normalized_smiles",
        }
    )

    # Remove rows without a usable structure identity.
    before = len(train)

    train = train.dropna(
        subset=[
            "normalized_smiles",
            "inchikey14",
            "molecular_formula",
        ]
    ).copy()

    print(f"  Removed missing structure identity: {before - len(train):,}")

    # Calculate theoretical neutral mass.
    print("  Calculating theoretical neutral masses...")

    train["neutral_mass"] = train["molecular_formula"].map(
        formula_to_mass
    )

    invalid_mass = train["neutral_mass"].isna().sum()

    if invalid_mass:
        warnings.warn(
            f"{invalid_mass:,} train rows have invalid molecular formulas."
        )

    train = train.dropna(subset=["neutral_mass"]).copy()

    train["source"] = "train"

    return train[
        [
            "normalized_smiles",
            "inchikey",
            "inchikey14",
            "molecular_formula",
            "neutral_mass",
            "source",
        ]
    ]


# ---------------------------------------------------------------------------
# Load COCONUT
# ---------------------------------------------------------------------------

def load_coconut():
    print("\n[2/5] Loading COCONUT...")

    # Only read the columns we actually need.
    coconut = pd.read_csv(
        COCONUT_PATH,
        usecols=[
            "canonical_smiles",
            "standard_inchi_key",
            "molecular_formula",
            "exact_molecular_weight",
        ],
        low_memory=False,
    )

    print(f"  COCONUT rows: {len(coconut):,}")

    coconut = coconut.rename(
        columns={
            "canonical_smiles": "normalized_smiles",
            "standard_inchi_key": "inchikey",
            "molecular_formula": "molecular_formula",
            "exact_molecular_weight": "neutral_mass",
        }
    )

    before = len(coconut)

    coconut = coconut.dropna(
        subset=[
            "normalized_smiles",
            "inchikey",
            "neutral_mass",
        ]
    ).copy()

    print(f"  Removed missing structure identity: {before - len(coconut):,}")

    # COCONUT has the full InChIKey, so derive the first block.
    coconut["inchikey14"] = (
        coconut["inchikey"]
        .astype(str)
        .str.split("-")
        .str[0]
    )

    coconut["neutral_mass"] = pd.to_numeric(
        coconut["neutral_mass"],
        errors="coerce",
    )

    invalid_mass = coconut["neutral_mass"].isna().sum()

    if invalid_mass:
        warnings.warn(
            f"{invalid_mass:,} COCONUT rows have invalid exact masses."
        )

    coconut = coconut.dropna(subset=["neutral_mass"]).copy()

    coconut["source"] = "coconut"

    return coconut[
        [
            "normalized_smiles",
            "inchikey",
            "inchikey14",
            "molecular_formula",
            "neutral_mass",
            "source",
        ]
    ]


# ---------------------------------------------------------------------------
# Deduplicate
# ---------------------------------------------------------------------------

def unify(train, coconut):
    print("\n[3/5] Unifying databases...")

    combined = pd.concat(
        [train, coconut],
        ignore_index=True,
    )

    print(f"  Combined rows: {len(combined):,}")

    # ---------------------------------------------------------------
    # First collapse duplicates WITHIN each source.
    #
    # This prevents multiple COCONUT records / train records for
    # the same chemical structure from unnecessarily surviving.
    # ---------------------------------------------------------------

    combined = combined.drop_duplicates(
        subset=["inchikey14"],
        keep="first",
    ).copy()

    print(f"  Unique inchikey14: {len(combined):,}")

    # ---------------------------------------------------------------
    # The above loses provenance information.
    #
    # Reconstruct whether each structure exists in train/COCONUT.
    # ---------------------------------------------------------------

    train_keys = set(train["inchikey14"].dropna())
    coconut_keys = set(coconut["inchikey14"].dropna())

    def get_source(key):
        in_train = key in train_keys
        in_coconut = key in coconut_keys

        if in_train and in_coconut:
            return "train+coconut"
        elif in_train:
            return "train"
        else:
            return "coconut"

    combined["source"] = combined["inchikey14"].map(get_source)

    combined["in_train"] = combined["inchikey14"].isin(train_keys)
    combined["in_coconut"] = combined["inchikey14"].isin(coconut_keys)

    return combined


# ---------------------------------------------------------------------------
# Final validation / sorting
# ---------------------------------------------------------------------------

def finalize(df):
    print("\n[4/5] Validating and sorting...")

    # Ensure numeric mass.
    df["neutral_mass"] = pd.to_numeric(
        df["neutral_mass"],
        errors="coerce",
    )

    before = len(df)

    df = df.dropna(
        subset=[
            "inchikey14",
            "neutral_mass",
            "normalized_smiles",
        ]
    ).copy()

    print(f"  Removed invalid final rows: {before - len(df):,}")

    # Sanity check for physically impossible masses.
    bad_mass = (
        (df["neutral_mass"] <= 0)
        | ~np.isfinite(df["neutral_mass"])
    )

    if bad_mass.any():
        print(f"  Removing invalid masses: {bad_mass.sum():,}")
        df = df.loc[~bad_mass].copy()

    # Sort for binary-search MS1 retrieval.
    df = df.sort_values(
        "neutral_mass",
        kind="mergesort",
    ).reset_index(drop=True)

    # Stable integer candidate ID.
    df.insert(
        0,
        "candidate_id",
        np.arange(len(df), dtype=np.int64),
    )

    # Final column order.
    df = df[
        [
            "candidate_id",
            "normalized_smiles",
            "inchikey",
            "inchikey14",
            "molecular_formula",
            "neutral_mass",
            "source",
            "in_train",
            "in_coconut",
        ]
    ]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CASMI UNIFIED CANDIDATE DATABASE BUILDER")
    print("=" * 70)

    train = load_train()
    coconut = load_coconut()

    unified = unify(train, coconut)

    final = finalize(unified)

    print("\n[5/5] Writing database...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    both = (final["source"] == "train+coconut").sum()
    train_only = (final["source"] == "train").sum()
    coconut_only = (final["source"] == "coconut").sum()

    print("\n" + "=" * 70)
    print("DATABASE COMPLETE")
    print("=" * 70)

    print(f"Output:              {OUTPUT_PATH}")
    print(f"Final candidates:    {len(final):,}")
    print()
    print(f"Train only:          {train_only:,}")
    print(f"COCONUT only:        {coconut_only:,}")
    print(f"Train + COCONUT:     {both:,}")
    print()
    print(
        f"Mass range:          "
        f"{final['neutral_mass'].min():.6f} - "
        f"{final['neutral_mass'].max():.6f} Da"
    )

    print("\nMissing values:")
    print(final.isna().sum())

    print("\nSource distribution:")
    print(final["source"].value_counts())

    print("\nFirst 10 candidates:")
    print(final.head(10).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()