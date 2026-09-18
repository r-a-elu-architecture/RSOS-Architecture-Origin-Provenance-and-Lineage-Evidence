# EXP-002 — V6 Correction and Scientific Authority

Date: 2026-09-18

Status: HISTORICAL ORIGINAL — RESPONSE VALIDITY COMPROMISED

Corrective successor: EXP-008, containing V6 Clean evidence.

## Original V6

Experiment: V6_STRUCTURAL_FINGERPRINT_VECTOR

- Reported successful cells: 8,000
- Empty extracted responses: 3,185
- Nonempty extracted responses: 4,815
- Confirmation AUC: approximately 0.944
- Final holdout AUC: approximately 0.930

Original interpretation:
STRONG_REPLICATED_STRUCTURAL_FINGERPRINT_VECTOR.

Preserve this interpretation as a historical result, not as
the final uncontaminated coordinate-level conclusion.

## Response-validity defect

The original runner could record an empty extracted response
as a successful observation.

Empty-response rates varied by experimental condition.

Consequently, the original numerical contrasts require
response-validity qualification.

## Corrective V6 Clean

Internal identifier:
V6_CLEAN_FINGERPRINT_CONFIRMATORY_REPLICATION

Location: EXP-008.

- Valid nonempty responses: 8,000
- Confirmation AUC: approximately 0.968
- Final holdout AUC: approximately 0.970

Clean coordinate findings:

- Direction: did not pass.
- Order: did not pass.
- Meta-control: passed.
- Preservation: passed.
- Terminal: passed.

Generic-recursion specificity did not pass.

## Correction limitations

The original and clean frozen prompt CSVs are byte-identical.

The corrective runner changed response extraction,
nonempty-response enforcement, retry handling and
API reasoning configuration.

The changes in results cannot be attributed solely
to one isolated modification.

The clean scalar metric uses lexical feature densities.
It does not directly encode graph-edge orientation
or temporal precedence.

Therefore, failed direction/order contrasts under
this metric do not establish that relational direction
or order has no effect under other valid measurements.

The generic-recursion control shares vocabulary
with the scoring features.

Its higher aggregate score is a failed specificity
contrast under the implemented measurement, not
evidence of RSOS-specific uniqueness.

The classifier AUC must not be described as a purely
relational or topology-specific measurement.

## Historical correction chain

September 10, 2026:
The user challenged experimental implementation and
omitted evidence. Response-validity contamination
was subsequently identified.

V6 Clean corrected response validity but produced
a narrower set of supported coordinates.

September 11, 2026:
The user challenged the recursion, complexity,
direction and order interpretations.

The subsequent methodological review identified
limitations in lexical scoring and control design.

V11 relational-graph testing was developed as a
separate metric repair.

## Scientific authority

Original V6:
Historical; coordinate-level interpretation superseded
where contradicted by valid corrective evidence.

V6 Clean:
Corrective, partially supported; metric-specificity
limitations remain.

V11:
Separate relational measurement investigation.

EXP-008's publication-folder name and internal V6 Clean
experiment identifier must remain explicitly cross-referenced.

## Preservation

This addendum does not replace or modify historical
protocols, runners, raw responses, outputs, manifests
or freeze records.

Cross-reference:
99_INDEX/INVESTIGATION_01_V5_V6_V6_CLEAN_LINEAGE.md
