# RSOS / RSSO / RSIA / RSX — Experimental Evidence

This repository is a curated publication derivative of the reconstructed
RSOS / RSSO / RSIA / RSX experimental evidence estate.

The original source and historical evidence remains preserved separately.
This repository is intended to provide a navigable, reproducible, and
auditable representation of the experiment sequence without rewriting the
underlying source estate.

## Repository structure

### `00_RECONSTRUCTION`

Master reconstruction control records and final reconstruction freezes.

### `01_CANONICAL_EXPERIMENTS`

Sixteen reconstructed canonical experiments.

Each experiment uses the common evidence structure:

- `00_CONTROL`
- `01_PROTOCOL`
- `02_INPUTS`
- `03_RUNNER`
- `04_RAW`
- `05_OUTPUTS`
- `06_DIAGNOSTIC`
- `07_ADJUDICATION`
- `08_PACKAGE`
- `09_FREEZE`

### `02_HISTORICAL_RUNS`

Ten preserved historical/developmental experiment records.

These records are retained because historical, incomplete, negative,
superseded, diagnostic, and developmental runs are part of the provenance
chain and must not be silently removed.

### `03_ACTIVE_EXPERIMENTS`

Current work that has not yet entered the closed canonical experiment chain.

### `90_ARCHIVE`

Preserved superseded/pre-execution branches.

### `99_INDEX`

Repository-wide provenance and publication controls, including:

- source-to-staging mapping
- copied evidence SHA-256 manifest
- experiment index
- local-artifact index
- publication preflight records
- technical copy-repair records

## Evidence policy

File presence alone does not establish scientific authority.

Interpretation of an experiment must follow its protocol, control,
adjudication, supersession, and freeze records.

Negative results are preserved.

Technical failures are distinguished from scientific negatives.

Incomplete and superseded runs are retained with their historical status.

Later reruns do not silently replace earlier evidence.

## Large and external artifacts

Some datasets, binary models, frozen archives, and very large artifacts are
not stored in ordinary Git history.

Their source locations and publication disposition are recorded in:

`99_INDEX/LOCAL_ARTIFACT_INDEX.csv`

This avoids treating omission from Git as omission from the evidence chain.

## Integrity

The ordinary publication evidence layer was copied from the reconstructed
canonical estate and verified against its source files using SHA-256.

The corresponding manifest is:

`99_INDEX/COPIED_STANDARD_EVIDENCE_SHA256.csv`

The original source estate is not modified by publication staging.

## Scientific correction records

- [Investigation 01: V5 → V6 → V6 Clean](99_INDEX/INVESTIGATION_01_V5_V6_V6_CLEAN_LINEAGE.md)
