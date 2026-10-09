"""Run the unified-database -> MS1 -> MS2 retrieval pipeline.

From the repository root:
    python runms2.py

By default this rebuilds data/candidate_db.csv from train.parquet and
coconutdb.csv, regenerates data/ms1_candidates.parquet, then ranks candidates
using matchms. To reuse an existing MS1 candidate file, pass
--reuse-ms1-candidates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from candidate_generation.ms2 import (
    load_candidate_reference_spectra,
    load_ms1_candidates,
    load_test_spectra,
    print_summary,
    rank_candidates,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT / "data" / "train.parquet"
DEFAULT_TEST = ROOT / "data" / "test.parquet"
DEFAULT_COCONUT = ROOT / "data" / "coconutdb.csv"
DEFAULT_CANDIDATES = ROOT / "data" / "ms1_candidates.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "ms2_ranked_candidates.parquet"


def run_preprocessing() -> None:
    """Rebuild the unified structure DB and regenerate MS1 candidates."""
    for label, path in (
        ("training Parquet", DEFAULT_TRAIN),
        ("test Parquet", DEFAULT_TEST),
        ("COCONUT CSV", DEFAULT_COCONUT),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}\n"
                "Place the required data file under the repository's data/ directory."
            )

    print("\n" + "=" * 68)
    print("STEP 1/3 — BUILD UNIFIED CANDIDATE DATABASE")
    print("=" * 68)
    subprocess.run(
        [sys.executable, str(ROOT / "data" / "unified_db_builder.py")],
        cwd=ROOT,
        check=True,
    )

    print("\n" + "=" * 68)
    print("STEP 2/3 — REGENERATE MS1 CANDIDATES")
    print("=" * 68)
    subprocess.run(
        [sys.executable, str(ROOT / "runms1.py")],
        cwd=ROOT,
        check=True,
    )

    if not DEFAULT_CANDIDATES.exists():
        raise FileNotFoundError(
            "MS1 pipeline completed without creating "
            f"{DEFAULT_CANDIDATES}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cosine-tolerance",
        type=float,
        default=0.01,
        help="Absolute m/z tolerance in Da for matchms cosine (default: 0.01).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65_536,
        help="Training Parquet rows per streaming batch.",
    )
    parser.add_argument(
        "--reuse-ms1-candidates",
        action="store_true",
        help="Skip database rebuilding and MS1 retrieval; reuse --candidates.",
    )
    args = parser.parse_args()

    custom_inputs = (
        args.train.resolve() != DEFAULT_TRAIN.resolve()
        or args.test.resolve() != DEFAULT_TEST.resolve()
        or args.candidates.resolve() != DEFAULT_CANDIDATES.resolve()
    )
    if custom_inputs and not args.reuse_ms1_candidates:
        parser.error(
            "Custom --train/--test/--candidates paths require "
            "--reuse-ms1-candidates. The automatic builder/MS1 stages use "
            "the canonical files under this repository's data/ directory."
        )

    for label, path in (("train", args.train), ("test", args.test)):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} file: {path}")

    if args.reuse_ms1_candidates:
        if not args.candidates.exists():
            raise FileNotFoundError(
                f"Missing MS1 candidate file: {args.candidates}"
            )
        print("Reusing existing MS1 candidates; database/MS1 rebuild skipped.")
    else:
        run_preprocessing()

    print("\n" + "=" * 68)
    print("STEP 3/3 — MS2 SPECTRAL-LIBRARY RETRIEVAL")
    print("=" * 68)
    print(f"Train:      {args.train}")
    print(f"Test:       {args.test}")
    print(f"Candidates: {args.candidates}")
    print(f"Output:     {args.output}")
    print(f"Cosine m/z tolerance: {args.cosine_tolerance:g} Da")

    test_spectra = load_test_spectra(args.test)
    candidates = load_ms1_candidates(args.candidates)
    candidate_keys = set(candidates["inchikey14"].astype(str))

    reference_library = load_candidate_reference_spectra(
        args.train,
        candidate_keys,
        batch_size=args.batch_size,
    )
    ranked = rank_candidates(
        test_spectra,
        candidates,
        reference_library,
        tolerance_da=args.cosine_tolerance,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(args.output, index=False)
    print_summary(ranked, reference_library)
    print(f"\nSaved ranked candidates to: {args.output}")
    print(
        "\nMS2 baseline complete. No MRR is reported because hidden test "
        "labels are unavailable."
    )


if __name__ == "__main__":
    main()
