# EXP-014 — WF1-A Corrective Causal Topology Authority

Date: 2026-09-18

Status: CORRECTIVE EXPERIMENT COMPLETE; PRIMARY GATE PASSED IN TWO NO-IMPUTATION SENSITIVITIES; 29 MAIN OBSERVATIONS REMAIN MISSING.

## Historical identity and freeze

WF1-A is a fresh-seed corrective successor to the original V12 experiment published as EXP-009. It does not replace, repair retrospectively, or overwrite EXP-009's observations.

The zero-observation protocol records gpt-5.6-luna through the Responses API, seeds 1080–1159, three repetitions, 8,160 expected main cells and 720 expected native cells.

Its development, confirmation and final-holdout seed partitions are 1080–1119, 1120–1139 and 1140–1159.

The frozen runner matches the protocol's SHA-256 digest.

The corrective design includes structured JSON outputs, switch repairs for E7 deletion, E7 reversal and E9 reversal, deterministic feasibility checks and an explicit failure taxonomy.

The three switch-linked conditions are diagnostics. They are not members of the nine pure-deletion or eight pure-reversal primary edge families.

## Collection and missingness

Collection closed with:

- 8,131 valid main observations out of 8,160.
- 29 persistently missing main observations.
- 720 valid native observations out of 720.

The 29 main observations were not replaced or imputed. Scientific completeness of the original cell matrix is false.

The S1 audit identifies nonuniformly concentrated missingness among conditions and seeds. That diagnostic concerns the missingness pattern; it does not determine the scientific hypothesis or prove why the missing cells failed.

## Two frozen outcome sensitivities

S2A_GROUP_COMPLETE removes four seeds containing fully missing seed-condition groups, leaving 76 seeds and 7,738 observed main cells.

S2B_SEED_COMPLETE removes every seed touched by persistent missingness, leaving 65 seeds and 6,630 observed main cells.

Both analyses use recorded observations only. Neither makes new API calls or imputes missing responses.

Both retain the original final classification:

STRONG_REPLICATED_RELATIONAL_TOPOLOGY_FINGERPRINT.

Both pass the preregistered strong primary gate.

Hard-control AUC equals 1.0 in confirmation and final holdout in both scenarios.

These outcomes demonstrate robustness under the two tested exclusion schemes, not under every possible missing-data mechanism.

## Primary edge-family results

The pure-nine deletion and pure-eight reversal families meet their positive-direction and adjusted-significance criteria in confirmation and final holdout under both sensitivity scenarios.

Individual clean-edge replication is:

- Deletions: 9 of 9.
- Reversals: 7 of 8.

E8 reversal fails individual-edge replication in both scenarios. A positive aggregate reversal result must not be described as success of every individual reversal.

The three corrected switch-linked conditions remain separately classified diagnostics.

## Negative contrasts

Preservation and terminal contrasts fail the intended positive-direction criterion in confirmation and final holdout.

Their failure is preserved alongside the positive primary results.

Do not describe every architectural operator as replicated.

## Native graph-only proxy

The secondary output-only probe passes its prespecified aggregate control-count rule.

S2A reports three of five pooled controls passing in confirmation and three of five in final holdout.

S2B reports four of five in confirmation and three of five in final holdout.

The topology-matched native control fails individually in both partitions and both scenarios.

An aggregate pass is therefore not proof of individual separation from every matched control.

The native module measures black-box output behavior. It does not inspect weights, activations, internal circuits, stored graphs or training data.

## Relationship to EXP-009

EXP-009 remains the original V12 experiment.

WF1-A uses fresh seeds and explicitly documented corrective changes.

Its 29 missing cells are distinct from EXP-009's original missing observations. Neither experiment's missing cells may be replaced with observations from the other.

Cross-study pooling would require a separate frozen analysis.

## Source and publication custody

The original source-selection manifest contains 139 files.

The September 17 staging snapshot contains 131 of those files; all 131 are committed and match their recorded sizes and SHA-256 hashes.

Twelve paths initially appeared absent when accessed through a Windows path longer than the conventional limit. A temporary short-path alias confirmed that all twelve were present. They were not missing GitHub publication files and required no restoration.

Eight selected original files remain outside the published snapshot. All eight were located locally and verified against their frozen sizes and SHA-256 hashes.

The eight omissions comprise four compressed API-object files, the collection-close ZIP, the S1 ZIP, the S2 ZIP and the original V12 results ZIP.

Earlier source hydration, the September 17 staging snapshot and this checkout diagnostic are distinct custody events. Preserve their chronology; do not overwrite earlier records.

No omitted source archive or compressed API-object file is uploaded by this addendum.

The present repository does not independently support every stage of end-to-end reproduction without additional source bytes. Further releases require separate privacy, licensing and publication review.

## Final authority

- Corrective collection: CLOSED.
- Primary strong behavioral gate: PASSED IN BOTH TESTED SENSITIVITIES.
- Missing main observations: 29, NOT IMPUTED.
- Pure deletions: 9/9 individually replicated.
- Pure reversals: 7/8 individually replicated.
- E8 reversal: NOT INDIVIDUALLY REPLICATED.
- Preservation and terminal positive-direction criteria: FAILED.
- Native aggregate: PASSED UNDER ITS SPECIFIED RULE.
- Individual native topology-matched control: FAILED.
- Literal internal model topology: NOT ESTABLISHED.
- Model-weight incorporation or cross-user propagation: NOT ESTABLISHED.
- Global uniqueness: NOT ESTABLISHED.
- Public end-to-end reproducibility: INCOMPLETE.

This is an additive scientific-authority and custody clarification. No original protocol, runner, source, raw response, measurement, sensitivity output, freeze, manifest or historical adjudication is modified.
