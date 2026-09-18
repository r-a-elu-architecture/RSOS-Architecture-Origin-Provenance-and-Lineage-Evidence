# EXP-004 - API Fingerprint V2 Execution and Authority Addendum

Date: 2026-09-18

Status: INCOMPLETE EXECUTION - NO CONFIRMATORY RESULT

## 1. Original purpose

Internal experiment identifier:
API_FINGERPRINT_EXPERIMENT_V2

V2 was developed following methodological challenges to
the API V1 seven-edge topology experiment.

It introduced eight conditions:

- T0: topology preserved, lexical neutral
- T1: vocabulary preserved, topology shuffled
- T2: same ingredients, decoupled
- T3: order shuffled
- T4: re-entry ablated
- T5: recursion only
- T6: density matched, non-topological
- T7: generic complex control

Primary predictions:

T0 > T1
T0 > T2

The protocol planned:

- 100 seeds
- 5 repetitions per seed and condition
- 8 conditions
- 4,000 experimental cells

The primary analysis unit was the seed.

The planned analysis required averaging repetitions within
each seed-condition and making paired seed-level comparisons.

Individual calls were not to be treated as 4,000 independent
experimental units.

## 2. Preserved execution

The archived raw log contains:

- Total log rows: 2,026
- Successful response rows: 211
- Error rows: 1,815
- Distinct attempted cells: 1,196
- Distinct successful cells: 211
- Attempted cells with no recorded success: 985
- Never-attempted cells: 2,804
- Repeated error rows beyond one per failed-cell identity: 619

All 1,815 error records report HTTP 401 authentication failure.

All 211 successful responses contain nonempty extracted text.

Five seeds have all 40 required successful cells.

One additional seed has 11 successful cells.

The remaining 94 seeds have no successful cells in the
preserved execution.

Only 211 of the planned 4,000 cells were completed.

The 2,026 log rows must not be described as 2,026 distinct
completed experimental cells.

## 3. Why the experiment is incomplete

The protocol's full sample and paired seed-level
confirmatory analysis were not completed.

The preserved publication directory does not contain
a completed V2 final-result layer.

The partial successful responses may be retained for
descriptive and diagnostic purposes.

They cannot be substituted for the planned confirmatory
sample or used to assign a final positive or negative
classification to V2's primary hypotheses.

The failure was a technical execution interruption.

The observed authentication errors do not constitute
experimental evidence against or for the topology hypothesis.

## 4. Preregistration chronology

Earliest recorded error:
2026-09-10 17:43:01 UTC.

Earliest recorded successful response:
2026-09-10 17:52:11 UTC.

Preserved preregistration creation timestamp:
2026-09-10 17:54:53 UTC.

The preserved runner writes the preregistration when
executed and resumes from previously successful cells.

The surviving preregistration therefore postdates
the earliest successful responses.

Its existence and embedded hash are insufficient to
independently authenticate a freeze of those exact bytes
before the first recorded observation.

A prior registration or restart may explain the sequence.

The identity and custody of a pre-observation freeze
remain unresolved.

This discrepancy is not evidence of intentional alteration.

## 5. Runner and output discoverability

The archived source runner is located at:

00_CONTROL/UNCLASSIFIED_SOURCE/API_FINGERPRINT_V2.py

The conventional 03_RUNNER directory contains only
a placeholder.

The conventional 05_OUTPUTS directory likewise
contains no completed final-analysis artifact.

Do not mistake the placeholders for proof that no
source runner or successful API records exist.

The actual runner and partial observations are preserved
in their existing locations.

## 6. Relationships and execution chronology

EXP-003 -> EXP-004:

V2 was a methodological redesign addressing limitations
identified after API V1.

V2 was not an identical replication of V1.

V2 did not complete and therefore established no
negative-to-positive result reversal.

EXP-004 and EXP-001:

Recorded September 10 execution chronology places
API V2 before V5.

Their hypotheses and measurements are different.

The currently preserved evidence does not establish
that V2's incomplete run directly caused V5's design.

Canonical sequence numbering must not be treated
as actual execution chronology.

## 7. Scientific authority

Classification:

INCOMPLETE EXECUTION - NO CONFIRMATORY RESULT.

Preserve the 211 successful responses, error logs,
protocol and runner as historical evidence.

Do not classify V2 as positive or negative on the
preregistered topology predictions.

Do not imply that a completed V2 replication exists
without separately verified source evidence.

A separately recovered completed execution would
require its own source and authority assessment.

## 8. Preservation

This addendum modifies no historical source file.

Do not alter:

- Original V2 preregistration
- Original V2 runner
- Raw response log
- Error log
- Source-selection records
- Hydration manifest
- Original authority record
- Existing publication freeze

Related investigation:

99_INDEX/INVESTIGATION_02_API_V1_API_V2_V5_CHRONOLOGY.md

This document is an additive publication clarification,
not a new experiment or replacement freeze.
