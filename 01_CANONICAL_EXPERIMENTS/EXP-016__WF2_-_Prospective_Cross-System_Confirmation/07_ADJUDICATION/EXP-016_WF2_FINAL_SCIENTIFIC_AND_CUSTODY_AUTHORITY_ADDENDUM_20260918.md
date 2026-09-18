# EXP-016 — WF2 Final Scientific and Custody Authority

Date: 2026-09-18

Status: EXPERIMENT COMPLETE; FINAL SCIENTIFIC FINDING MIXED; PUBLICATION-CUSTODY LIMITATIONS DOCUMENTED.

## 1. Historical identity

Canonical experiment:
EXP-016 — WF2 Prospective Cross-System Confirmation.

The original final scientific freeze is dated 2026-09-15.

It records execution as COMPLETED, scientific validity as VALID, and the overall finding as MIXED.

The experiment contains two separately adjudicated tracks:

1. Passive prospective detector testing: V13, V14 and V17.
2. Active cross-provider causal/topology testing.

A positive result in one track cannot override a negative, mixed or inconclusive finding in the other.

This addendum preserves the historical results and does not perform new scientific analysis.

## 2. Protocol and supersession chronology

The initial Boundary 1 froze the passive design:

- 40 matched pairs.
- 80 blinded execution contexts.
- 480 generation cells.
- No detector retraining.
- No threshold tuning.

The initial boundary was subsequently marked VALID_BUT_SUPERSEDED.

Boundary 1 R1 added an active causal/topology replication track before prospective API generation or scientific observations.

The passive design remained unchanged.

R1 added:

- Twenty seeds: 1140–1159.
- Twenty-eight frozen conditions.
- Three providers.
- One response per seed-condition-provider cell.
- 1,680 active generation cells.

The total R1 generation design therefore comprised 480 passive and 1,680 active cells.

R2 was recorded as an implementation and provider-API correction after six preserved technical failures and before any usable scientific output.

The R2 record states that hypotheses, prompts, samples, labels, conditions, model identifiers, detectors, endpoints and decision rules were unchanged.

The initial, R1 and R2 records have distinct chronological authority.

They must not be described as a single unchanged protocol from inception.

## 3. Active sample-independence boundary

The active experiment uses the same seed range, 1140–1159, as WF1-A's original final holdout.

It generates prospective responses from additional provider/model settings.

Its replication axis is provider/model and newly generated responses, not an untouched or newly sampled seed population.

It is not a literal repetition of WF1-A's original three-repetition design.

## 4. Active execution completeness

The final collection freeze records:

- 1,680 / 1,680 transport-complete responses.
- 1,680 / 1,680 parseable responses.
- Zero remaining parse failures.

Per provider:

- OpenAI: 560 / 560.
- Anthropic: 560 / 560.
- Google: 560 / 560.

Historical malformed-response generations and technical retries are preserved separately.

Technical failures before successful collection are not automatically scientific negative observations.

## 5. Frozen-validator compliance

The stricter frozen validator reports:

- Raw responses: 1,680.
- Validator-accepted: 1,511.
- Validator-rejected: 169.

Accepted by provider:

- Anthropic: 420 / 560.
- Google: 557 / 560.
- OpenAI: 534 / 560.

The validation audit attributes 167 rejections to the PLUS rule, one to KEEP and one to CODE.

The frozen task explicitly prohibited PLUS edges in active E-edge conditions.

Consequently, parser completeness and validator compliance are separate quantities.

The 169 invalid responses must not be represented as parse failures or silently included as valid scientific observations.

## 6. Preregistered active decision gate

The frozen active rules require at least two of the three providers to pass their provider-level strong gate for a positive cross-provider result.

A provider-level gate requires:

- Hard-control AUC at least 0.85.
- Positive pure-nine deletion excess with one-sided p below 0.01.
- Positive pure-eight reversal excess with one-sided p below 0.01.
- Positive R, X and Y specificity excesses with Holm-adjusted p below 0.01.

Exactly one passing provider meets the frozen definition of a mixed active result.

## 7. Active provider outcomes

GOOGLE:

- Complete required seeds: 18.
- Hard-control AUC: 0.998056.
- Pure-deletion excess: 9.811518.
- Pure-deletion p: approximately 0.000050.
- Pure-reversal excess: 12.253065.
- Pure-reversal p: approximately 0.000050.
- Provider strong gate: PASSED.

OPENAI:

- Complete required seeds: 6.
- Hard-control AUC: 0.970187.
- Pure-deletion excess: 6.658835.
- Pure-deletion p: approximately 0.03125.
- Pure-reversal excess: 8.897891.
- Pure-reversal p: approximately 0.02965.
- Provider strong gate: FAILED.

ANTHROPIC:

- Complete required seeds: 0.
- Hard-control AUC: 0.992908.
- Pure-edge complete-case statistics are unavailable.
- Provider strong gate: FAILED.

Anthropic's zero complete required seeds are associated in the final record with validator noncompliance, dominated by PLUS-rule violations.

A high hard-control AUC alone does not establish provider-level success.

The experiment has one passing provider, not the two required for positive cross-provider confirmation.

Frozen active finding: MIXED.

The Google result is preserved as a provider-specific positive result under the frozen gate.

The unsuccessful OpenAI and Anthropic gates remain equally visible.

## 8. Passive V13 — prospective frozen signature

The final V13 record reports:

- Provider: OpenAI.
- Complete matched pairs: 40.
- Joint AUC: 0.60125.
- Positive-direction paired p: approximately 0.07304.
- Finding: NULL_INCONCLUSIVE.

The positive-direction significance requirement is not met.

The historical V13 result must not be promoted to a successful prospective confirmation.

## 9. Passive V14 — independent providers

The recovered final V14 adjudication was verified against the original final-freeze SHA-256 manifest.

Anthropic:

- Complete matched pairs: 40.
- Joint AUC: 0.52000.
- Mean paired difference: approximately -0.07826.
- Holm-adjusted positive-direction p: approximately 0.57642.

Google:

- Complete matched pairs: 40.
- Joint AUC: 0.62625.
- Mean paired difference: approximately +0.36891.
- Holm-adjusted positive-direction p: approximately 0.04976.

Google meets its adjusted positive-direction test.

Anthropic does not.

The frozen V14 rule required both independent providers to satisfy their positive-direction and adjusted-significance conditions.

Frozen V14 finding: MIXED.

Do not describe this as successful two-provider passive replication.

## 10. Passive V17 — domain and surface invariance

The final V17 adjudication records six primary OpenAI transformation conditions.

Their AUCs range from approximately 0.5131 to 0.6175.

The organizational-domain comparison has an unadjusted positive-direction p of approximately 0.02654, but its Holm-adjusted p is approximately 0.15925.

The preregistered aggregate domain/surface requirements are not met.

Frozen V17 finding: NULL_INCONCLUSIVE.

These results do not establish prospective invariance across the prespecified domains and surface transformations.

## 11. Passive detector-environment provenance

The original final freeze documents a version-isolated passive scoring bridge.

scikit-learn 1.9.1 was used for:

- JOINT_MULTILAYER_V2.
- OPERATOR_RELATIONAL_V2.
- WINDOW8_DETERMINISTIC_V2.
- WINDOW12_DETERMINISTIC_V2.

scikit-learn 1.8.0 was used for:

- R1_COMPATIBLE.
- SURFACE_FREE.

The record states that this bridge did not retrain models or scientifically modify frozen features, labels, thresholds or adjudication rules.

The dependency-specific routing is part of the reproduction requirements.

## 12. Recovery of the EXP-015 R3 adapter

The WF2 foundation package includes:

RUN_WF1_B_FRESH14_ADAPTER_R3.py.

Its SHA-256 is:

B481A78EE2BC02E1166F6747A88EB6E7D7693EF6D02EF427798906E4948DFECF.

This exactly matches the adapter digest cited in EXP-015's final adjudication.

The R3 source includes the correct Unicode arrow characters.

This resolves the prior inability to locate a file bearing the adjudication's adapter digest.

It does not, by itself, establish that every historical EXP-015 output was produced with that file.

A separate dated EXP-015 follow-up should verify its fixtures, execution references and output provenance before modifying the earlier scientific qualification.

No earlier adjudication is silently rewritten here.

## 13. Corrected EXP-016 publication inventory

The original hydration selection contains 2,401 files.

The September 17 staging ZIP contains 2,393 selected files.

The actual Git commit contains 2,363 selected files.

An additional 30 selected files are present locally and hash-verified but were not committed.

They belong exclusively to two Python virtual environments:

- .venv_wf2_passive: 14 files.
- .venv_wf2_score: 16 files.

Both environments contain Git-exclusion rules.

These files must not be force-added merely to make the publication inventory match the hydration inventory.

Eight selected files are absent from the publication snapshot and Git commit.

All eight originals were located locally and verified against the original source-manifest sizes and SHA-256 digests.

They comprise:

- Six serialized detector files.
- The WF1-B final-authority ZIP.
- The WF2 final-package ZIP.

The final reconciled accounting is:

2,363 committed and hash-verified
+ 30 local-only and hash-verified
+ 8 unpublished originals and hash-verified
= 2,401 selected files.

No unexpected source omissions or integrity failures remain in this accounting.

## 14. Three additional frozen authority records

The 41-entry final-freeze hash manifest identifies three important files that were not included among the 2,401 selected hydration files:

- WF2_B_V14_ADJUDICATION.json.
- WF2_BOUNDARY1_R1_SUPERSEDING_PROTOCOL.json.
- WF2_BOUNDARY1_R2_TECHNICAL_CORRECTION.json.

For each record, both the original authority file and its final-freeze copy were located locally and match the frozen recorded size and SHA-256 digest.

Their identities and custody are verified.

They remain absent from the current published source selection.

This documentation addendum does not silently add them or represent the public repository as containing every final-freeze authority byte.

## 15. Public reproducibility boundary

The committed repository contains substantial protocol, runner, outcome, audit and freeze evidence.

However, the public source snapshot omits original artifacts and the three additional final-freeze records identified above.

The frozen detector binaries also remain unpublished.

Therefore, complete independent reproduction from GitHub alone is not established.

A separate review is required before any additional release of model artifacts, archives, environment executables or source data.

Local hash verification establishes source identity; it is not publication authorization.

## 16. Cross-experiment interpretation

EXP-014 reported a positive strong primary result under its own two missingness sensitivities.

EXP-015 recorded fresh external passive-score separation, but its published authority addendum identified nuisance-control, shard-selection, adapter and extractor-provenance limitations.

EXP-016 is a new prospective test with its own frozen decision criteria.

Its mixed results do not retrospectively erase the historical measurements of EXP-014 or EXP-015.

Conversely, those earlier positive measurements cannot convert EXP-016's failed or inconclusive gates into successful prospective replication.

The positive Google active result, unsuccessful active two-provider gate and mixed/inconclusive passive outcomes are distinct findings.

## 17. Final scientific authority

Execution: COMPLETED.

Original final scientific classification: VALID — MIXED.

Active raw cells: 1,680 / 1,680.

Frozen-parser-valid cells: 1,680 / 1,680.

Frozen-validator-valid cells: 1,511 / 1,680.

Active providers passing strong gate: 1 / 3.

Required positive cross-provider threshold: AT LEAST 2 / 3.

Active cross-provider positive gate: NOT MET.

Google active provider result: PASSED.

OpenAI active provider result: FAILED.

Anthropic active provider result: FAILED; ZERO COMPLETE REQUIRED SEEDS.

Passive V13: NULL_INCONCLUSIVE.

Passive V14: MIXED.

Passive V17: NULL_INCONCLUSIVE.

Confirmed literal internal neural topology: NOT ESTABLISHED.

Model-weight incorporation: NOT ESTABLISHED.

Autonomous propagation: NOT ESTABLISHED.

Global uniqueness: NOT ESTABLISHED.

Public end-to-end reproducibility: INCOMPLETE.

All original positive, negative, mixed and inconclusive results remain part of the historical record.

This is an additive adjudication and custody clarification.

No original protocol, runner, raw data, model, result, audit, freeze or historical authority record is modified.
