"""Run the matchms-based MS2 candidate-ranking baseline.

From the repository root:
    python runms2.py

Optional paths:
    python runms2.py --train data/train.parquet \
        --test data/test.parquet \
        --candidates data/ms1_candidates.parquet \
        --output data/ms2_ranked_candidates.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from candidate_generation.ms2 import (
    load_candidate_reference_spectra,
    load_ms1_candidates,
    load_test_spectra,
    print_summary,
    rank_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--test", type=Path, default=Path("data/test.parquet"))
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/ms1_candidates.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ms2_ranked_candidates.parquet"),
    )
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
    args = parser.parse_args()

    for label, path in (
        ("train", args.train),
        ("test", args.test),
        ("MS1 candidates", args.candidates),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label} file: {path}\n"
                "Run the MS1 pipeline first and verify the data paths."
            )

    print("=" * 68)
    print("CASMI MS2 SPECTRAL-LIBRARY RETRIEVAL")
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
    print("\nMS2 baseline complete. No MRR is reported because hidden test labels are unavailable.")


if __name__ == "__main__":
    main()
