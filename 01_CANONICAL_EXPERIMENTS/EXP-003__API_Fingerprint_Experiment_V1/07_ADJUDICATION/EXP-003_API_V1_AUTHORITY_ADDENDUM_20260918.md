# EXP-003 - API Fingerprint V1 Authority Addendum

Date: 2026-09-18

Status: HISTORICAL NEGATIVE - RESPONSE VALIDITY COMPROMISED;
PREREGISTRATION CHRONOLOGY UNRESOLVED.

## Original experiment

API_FINGERPRINT_EXPERIMENT_V1 tested whether authentic,
partially blinded RSOS topology produced a higher frozen
seven-edge kernel score than topology-shuffled and
same-ingredients-decoupled controls.

The original report records 5,000 nominally successful API calls.

Original mean kernel scores:

- Authentic: 1.196
- Topology shuffled: 1.336
- Same ingredients decoupled: 1.208
- K6 ablation: 1.118
- Generic complex control: 0.688

The primary authentic-topology superiority prediction
was not supported under the original measurement.

The authentic condition exceeded the generic-complex
control but not the matched topology controls.

## Response-validity defect

A subsequent raw-response audit identified:

- Total nominal successes: 5,000
- Empty extracted responses: 881
- Nonempty responses: 4,119

Empty responses by condition:

- Authentic: 190 / 1,000
- Topology shuffled: 253 / 1,000
- Decoupled: 138 / 1,000
- K6 ablation: 228 / 1,000
- Generic complex: 72 / 1,000

The original runner allowed an empty extracted response
to be scored and recorded as a successful observation.

Missingness is condition-dependent.

Original numerical estimates therefore require
response-validity qualification.

Exploratory nonempty-only means:

- Authentic: approximately 1.477
- Shuffled: approximately 1.789
- Decoupled: approximately 1.401

These are descriptive post-hoc diagnostics, not
replacement confirmatory results. Filtering on
condition-dependent response validity may introduce bias.

## Measurement limitations

The detector scores motif-family combinations in
generated prose.

Motif co-occurrence is not independent verification
of the directed relations between those motifs.

Historical longitudinal Top100 observations and
fresh stateless API outputs are different populations.

The original call-level tests also do not account
adequately for repeated observations within seeds.

A one-sided test in the preregistered positive
direction must not be interpreted as demonstrating
equality when the observed contrast is reversed.

The original K6 ablation summary does not establish
a selective causal effect on K6 and K7 individually.

## Preregistration chronology discrepancy

Preserved raw-response timestamps:
2026-09-10 15:54:36 to 16:23:53 UTC.

Preserved preregistration creation timestamp:
2026-09-10 16:34:47 UTC.

Preserved pre-API hash-lock timestamp:
2026-09-10 16:34:47 UTC.

The original runner writes the preregistration and
hash-lock files when executed, without a demonstrated
preservation guard against regeneration.

Consequently, the surviving files do not independently
establish that their exact bytes were frozen before
the earliest recorded API responses.

A prior freeze may have existed. Its identity and
pre-observation custody remain unresolved.

This is a provenance discrepancy, not evidence of
intentional alteration.

## Historical correction sequence

September 10, 2026:

16:32 UTC:
The assistant interpreted API V1 as failing its
specific topology-superiority prediction.

16:37 UTC:
The user challenged omitted historical information
and requested examination of the wider evidence.

16:52 UTC:
The assistant acknowledged measurement and
population-comparability discrepancies.

17:39-17:41 UTC:
API V2 was proposed as a redesigned experiment
with stronger controls and seed-level paired analysis.

The published V2 branch was not completed.

## Relationship to EXP-004 and EXP-001

EXP-003 -> EXP-004:
Confirmed methodological redesign.

EXP-004 did not produce a complete confirmatory
result in the preserved published execution.

No negative-to-positive reversal is established.

EXP-003 preceded EXP-001 in recorded execution time.

Their research questions differ. V5 is not an
identical clean replication of API V1.

## Final scientific authority

Preserve the original negative outcome for the
specific V1 topology-superiority prediction.

Do not generalize it into falsification of the
historical structural architecture.

Do not describe the generic-complex result as
RSOS-specific topology confirmation.

Do not treat incomplete API V2 as a positive result.

The original files and historical interpretations
remain preserved unchanged.

Related publication record:
99_INDEX/INVESTIGATION_02_API_V1_API_V2_V5_CHRONOLOGY.md

This addendum is a documentation correction,
not a new experiment or a replacement freeze.
