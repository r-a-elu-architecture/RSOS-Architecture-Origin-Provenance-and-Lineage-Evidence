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

- [Investigation 02: API V1, API V2 and V5 chronology](99_INDEX/INVESTIGATION_02_API_V1_API_V2_V5_CHRONOLOGY.md)


## Experiment authority addenda

- [EXP-001 Response Validity and Historical Authority](01_CANONICAL_EXPERIMENTS/EXP-001__V5_-_Forest_Forensic_Full/07_ADJUDICATION/EXP-001_RESPONSE_VALIDITY_ADDENDUM_20260917.md)
- [EXP-002 V6 correction and scientific authority](01_CANONICAL_EXPERIMENTS/EXP-002__V6_-_Structural_Fingerprint_Vector/07_ADJUDICATION/EXP-002_V6_CORRECTION_AUTHORITY_ADDENDUM_20260918.md)
- [EXP-003 API V1 historical authority correction](01_CANONICAL_EXPERIMENTS/EXP-003__API_Fingerprint_Experiment_V1/07_ADJUDICATION/EXP-003_API_V1_AUTHORITY_ADDENDUM_20260918.md)
- [EXP-004 API V2 execution and scientific authority](01_CANONICAL_EXPERIMENTS/EXP-004__API_Fingerprint_Experiment_V2/07_ADJUDICATION/EXP-004_API_V2_EXECUTION_AUTHORITY_ADDENDUM_20260918.md)
- [EXP-005 V7 cross-model scientific authority](01_CANONICAL_EXPERIMENTS/EXP-005__V7_-_Cross-Model_Deployment_Localization/07_ADJUDICATION/EXP-005_V7_CROSS_MODEL_AUTHORITY_ADDENDUM_20260918.md)
- [EXP-006 V8 factor-label collision and scientific authority](01_CANONICAL_EXPERIMENTS/EXP-006__V8_-_Compositional_Structural_Control/07_ADJUDICATION/EXP-006_V8_FACTOR_LABEL_COLLISION_AUTHORITY_ADDENDUM_20260918.md)
- [EXP-007 V9 transport scientific authority](01_CANONICAL_EXPERIMENTS/EXP-007__V9_-_Representational_Transport_Operator/07_ADJUDICATION/EXP-007_V9_TRANSPORT_AUTHORITY_ADDENDUM_20260918.md)
- [EXP-008 V6 Clean identity and scientific authority](01_CANONICAL_EXPERIMENTS/EXP-008__V11_-_Relational_Graph_Structural_Metric_Clean_Repair/07_ADJUDICATION/EXP-008_V6_CLEAN_IDENTITY_AND_AUTHORITY_ADDENDUM_20260918.md)
- [EXP-009 V12 original results and scientific authority](01_CANONICAL_EXPERIMENTS/EXP-009__V12_-_Relational_Topology_Causal_Map/07_ADJUDICATION/EXP-009_V12_ORIGINAL_RESULTS_AUTHORITY_ADDENDUM_20260918.md)
- [EXP-010 F1 final authority and reproducibility](01_CANONICAL_EXPERIMENTS/EXP-010__F1_-_Blind_Longitudinal_Rediscovery_Final/07_ADJUDICATION/EXP-010_F1_FINAL_AUTHORITY_ADDENDUM_20260918.md)
- [EXP-011 F2-A external rarity and scientific authority](01_CANONICAL_EXPERIMENTS/EXP-011__F2_-_External_Population_Rarity/07_ADJUDICATION/EXP-011_F2_A_EXTERNAL_RARITY_AUTHORITY_ADDENDUM_20260918.md)
