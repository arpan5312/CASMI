# CASMI
# CASMI Production — Approach & Architecture

## Objective

Predict up to 25 candidate SMILES for each CASMI test molecule from LC-MS/MS spectra.

Primary metric: **MRR@25**, evaluated by matching the first InChIKey block (`inchikey14`) after the competition's specified RDKit tautomer canonicalization.

The central design principle is:

> **Extract deterministic chemical information first. Use ML to resolve uncertainty, not to replace known chemistry.**

---

## Core Architecture

```text
MS/MS spectrum + metadata
          │
          ▼
   Adduct / ion chemistry
          │
          ▼
      Neutral mass
          │
          ▼
   ┌───────────────────┐
   │      Model 1      │
   │ Predict mass      │
   │ search window ΔM  │
   └───────────────────┘
          │
          ▼
 Mass-based candidate retrieval
          │
          ▼
 Chemistry + spectral evidence
          │
          ▼
   ┌───────────────────┐
   │      Model 2      │
   │ Candidate ranker  │
   └───────────────────┘
          │
          ▼
        Top 25
```

This is a **retrieval-first, generation-second** system.

The candidate generator establishes the ceiling of the system: if the correct structure never enters the candidate set, no ranker can recover it.

---

# Phase 1 — Adduct / Ion Interpretation

The precursor `m/z` is not the neutral molecular mass.

We first interpret:

* `adduct`
* charge
* multiplicity
* added/subtracted chemical groups
* electron/proton corrections

Then calculate:

```text
neutral_mass = f(precursor_mz, adduct)
```

The parser should be general rather than relying on a tiny hard-coded lookup table because the dataset contains many adduct forms.

Special cases such as `[Cat]2+` and malformed/pathological records must be handled explicitly.

### Validation requirement

Neutral-mass calculation needs independent validation.

We should compare multiple calculation paths and inspect failures rather than silently accepting every row.

Dataset anomalies are possible; therefore:

> **A mathematically correct parser does not imply a chemically correct dataset.**

---

# Phase 2 — Neutral-Mass Candidate Retrieval

Once the target neutral mass is known, structures can be retrieved from a large chemical/spectral database using mass.

The basic structure:

```text
candidate neutral masses
        ↓
sort once
        ↓
binary/range search
        ↓
mass window
        ↓
candidate structures
```

This avoids repeatedly scanning millions of rows.

The training spectral library can provide a highly valuable retrieval source because it contains actual MS/MS observations rather than only theoretical structures.

A separate chemical database such as **COCONUT** can provide additional structural coverage, particularly for natural products.

However:

> More database coverage is not automatically better.

Additional candidates can dilute the ranking problem and introduce decoys. Candidate recall and final ranking quality must therefore be measured separately.

---

# Phase 3 — Model 1: Adaptive Mass Search Window

A fixed mass window is an obvious baseline, but it should not be treated as chemically guaranteed.

Instead, investigate a model that directly predicts the numerical search radius:

```text
target neutral mass
        +
spectrum / metadata
        ↓
      Model 1
        ↓
      ΔM
        ↓
[M − ΔM, M + ΔM]
```

The model should output an actual number, not an abstract category such as "narrow", "medium", or "broad".

### Training target

For held-out queries, determine how large a search radius is required to retrieve the true structure.

For example:

```text
5 Da   → truth absent
10 Da  → truth absent
20 Da  → truth absent
35 Da  → truth present
```

This can produce a target such as:

```text
ΔM = 35 Da
```

However, the final target should ideally be chosen using an explicit tradeoff between:

* truth retrieval/recall
* number of candidates
* downstream MRR

The goal is not simply to maximize the window.

### Possible future extension

Instead of one symmetric radius:

```text
[M − ΔM, M + ΔM]
```

the model could eventually predict:

```text
[M − ΔM_lower, M + ΔM_upper]
```

But the scalar-radius version is the first experiment.

---

# Phase 4 — Spectral / Chemical Evidence

Mass narrows the search but does not uniquely identify a molecule.

Isomers and isobars can share the same or extremely similar neutral masses.

The MS/MS spectrum provides additional information through:

* fragment `m/z`
* fragment intensities
* peak overlap
* spectral similarity
* fragmentation patterns
* precursor/adduct compatibility
* collision energy
* instrument type

`ms2_mzs` represents the observed fragment-ion m/z values.

`ms2_normalized_intensities` represents their relative intensities.

Together they form the observed MS/MS fingerprint.

---

# Fragment Chemistry / Neutral Loss

A potentially powerful future layer is explicit fragmentation reasoning.

Conceptually:

```text
precursor neutral mass
        │
        ├──── fragment ion
        │
        ▼
   possible neutral loss
```

For a correctly interpreted fragment:

```text
neutral loss ≈ precursor neutral mass − fragment neutral mass
```

But fragment interpretation is not trivial.

A fragment does **not necessarily have the same ion/adduct representation as the precursor**.

For example:

* protonation can redistribute
* sodium can remain attached to a fragment or be lost
* fragment charge can differ
* different ion formulas can produce the same observed m/z

Therefore we must not blindly apply the precursor adduct parser to every MS2 peak.

A future chemistry layer should instead consider multiple plausible fragment-ion representations and ask which explanations are chemically consistent with candidate structures.

This is conceptually similar to the chemistry-heavy reasoning used by systems such as SIRIUS/CSI:FingerID, but our implementation must be designed around the CASMI dataset and competition constraints.

---

# Phase 5 — Model 2: Candidate Ranker

After retrieval, Model 2 decides which candidates deserve the highest positions.

The learning problem becomes:

```text
(spectrum, candidate)
        ↓
     features
        ↓
      ranker
        ↓
candidate score
```

Potential features include:

### Mass evidence

* neutral-mass error
* precursor m/z error
* distance from predicted mass window

### Direct spectral evidence

* spectral similarity
* fragment peak overlap
* intensity similarity
* matched peak count
* similarity under m/z tolerance

### Analog evidence

* shifted spectral similarity
* mass difference to known structures
* analog similarity score

### Metadata/context

* adduct compatibility
* ionization mode
* collision energy
* instrument type
* library/source
* number/quality of available reference spectra

The ranker is where ML is most naturally useful:

> deterministic chemistry generates plausible hypotheses; ML learns how to combine imperfect evidence.

---

# Retrieval Classes

A useful conceptual separation is:

## Class 1 — Direct Library Match

The same/near-identical neutral-mass structure exists with an observed library spectrum.

Compare the query spectrum directly against reference spectra.

This is the strongest retrieval situation.

## Class 2 — Spectral Analog / Mass-Shift Match

The exact target structure is absent, but a related compound may have a useful spectrum.

If the target and known compound differ by a mass shift, fragments can sometimes be aligned after accounting for that shift.

Conceptually:

```text
target precursor
       │
       │ mass difference
       ▼
known analog
       │
       ▼
shifted fragment comparison
```

This can retrieve useful structural analog evidence even when there is no exact library hit.

## Class 3 — Hard / Unseen Case

Neither direct library evidence nor useful analog evidence is sufficient.

This is where more explicit chemistry and/or generative ML becomes important.

The system should not force every query through the same mechanism if the available evidence differs fundamentally.

---

# Representative Spectra

For structures with multiple reference spectra, using every spectrum can actually hurt discrimination.

A reference structure can be represented by a selected representative spectrum, initially the richest spectrum by peak count.

Reason:

```text
more spectra per structure
        ↓
more chances for irrelevant structures to obtain
a high accidental similarity
        ↓
flatter candidate scores
        ↓
worse discrimination
```

This is an empirical hypothesis and must be validated rather than assumed universally.

---

# Candidate-Cap Principle

Candidate caps are runtime safeguards, not modeling assumptions.

A cap that arbitrarily removes candidates can destroy recall.

Therefore:

```text
candidate generation
       ↓
measure truth retention
       ↓
then apply runtime cap if necessary
```

The important metric here is:

> **What fraction of queries still contain the truth after candidate generation?**

Not merely how fast retrieval runs.

---

# Validation Philosophy

The system must distinguish three different questions:

### 1. Retrieval Recall

```text
Did the correct structure enter the candidate set?
```

### 2. Ranking Quality

```text
Given that the truth was retrieved,
did the ranker place it highly?
```

### 3. End-to-End MRR@25

```text
Did the final system put the truth in the top 25,
and how high?
```

These must not be conflated.

A ranking model cannot fix candidate-generation failure.

---

# Experimental Discipline

Every important architectural decision should be experimentally justified.

For each change, measure at least:

```text
candidate recall
candidate count
MRR@25
runtime
```

Prefer query-level holdout validation.

Avoid in-sample experiments where the model is trained on the same query groups used to evaluate the change.

Important principle:

> **A local improvement is not evidence of a real improvement unless the experiment is designed to measure generalization.**

Hyperparameters should be treated as hypotheses, not truths.

Examples:

* mass-window size
* analog-window size
* number of analog candidates
* representative-spectrum strategy
* candidate cap
* similarity weighting
* ranker complexity

---

# Current Design Philosophy

The project is not:

```text
raw spectrum → giant neural network → SMILES
```

It is:

```text
physical/chemical interpretation
        ↓
deterministic mass inference
        ↓
intelligent retrieval
        ↓
spectral/chemical evidence
        ↓
learned ranking
        ↓
top-25 uncertainty-aware prediction
```

The system should exploit every piece of information that can be derived reliably before asking ML to make an empirical judgment.

---

# Open Research Questions

1. How accurately can precursor neutral mass be recovered across all adduct types?
2. How much candidate recall is obtained at different neutral-mass windows?
3. Can Model 1 predict an effective search radius better than a fixed window?
4. What observable features allow Model 1 to predict the required radius?
5. How much does COCONUT improve true candidate coverage?
6. At what point does additional database coverage become harmful dilution?
7. Which spectral similarity representation best separates isobars/isomers?
8. Does explicit neutral-loss reasoning improve ranking?
9. Can fragment-ion/adduct hypotheses be inferred reliably enough for chemistry-based scoring?
10. How much does instrument/collision-energy metadata improve spectral comparison?
11. When should Class 1, Class 2, and Class 3 mechanisms be invoked?
12. Can a ranker combine deterministic chemistry and spectral evidence without overfitting?
13. How much candidate recall is lost from any runtime cap?
14. What is the actual upper bound imposed by indistinguishable MS/MS spectra?

---

# North Star

Build a system that understands the information content of the experiment as deeply as possible.

Use:

**chemistry → retrieval → spectral evidence → ML**

rather than:

**ML → hope.**


---

# MS2 spectral-library baseline

The MS2 baseline ranks each molecule's MS1-retrieved candidate structures by the
maximum `matchms` greedy cosine similarity against training spectra with the
same `inchikey14`. Training Parquet is streamed in batches; only spectra for
candidate keys are retained in memory.

Install the additional dependencies in your project environment:

```bash
pip install -r requirements-ms2.txt
```

Run from the repository root:

```bash
python runms2.py
```

By default, `runms2.py` runs the full pipeline in order:

1. Rebuild `data/candidate_db.csv` from `data/train.parquet` and `data/coconutdb.csv`.
2. Regenerate `data/ms1_candidates.parquet` using `runms1.py`.
3. Run MS2 spectral retrieval and write `data/ms2_ranked_candidates.parquet`.

The builder and MS1 runner resolve paths relative to the repository, not the
terminal's current directory. To intentionally reuse an existing MS1 candidate
file, pass `--reuse-ms1-candidates`.

The ranked output includes `ms2_max_cosine`, the best absolute MS1 mass error
(used only as a tie-breaker), reference-spectrum coverage, rank, and an
`is_top25` flag. The scoring stage reports the number of cosine calls, positive
scores, and scoring exceptions. If every completed score is zero, treat that run
as a diagnostic failure rather than a useful ranking.

This is a baseline, not a validated MRR estimate. Measure it on a leakage-safe
molecule-level holdout before treating its ranking quality as evidence of hidden
test performance.
