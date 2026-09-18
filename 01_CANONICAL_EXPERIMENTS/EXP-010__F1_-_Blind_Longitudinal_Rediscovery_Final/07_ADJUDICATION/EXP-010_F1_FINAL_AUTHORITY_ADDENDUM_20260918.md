# EXP-010 - F1 Final Authority and Reproducibility Addendum

Date: 2026-09-18

Status:
FINAL HISTORICAL REDISCOVERY RESULT PRESERVED;
EXTERNAL POPULATION SEPARATION OBSERVED;
STRICT BR2 MATCHING FAILED;
PUBLICATION REPRODUCIBILITY INCOMPLETE.

## 1. Original classification

Preserved F1 final classification:

STRONG BLIND HISTORICAL STRUCTURAL REDISCOVERY
WITH INDEPENDENT LM-ARENA POPULATION SEPARATION.

The original adjudication, status ledger and claim
ladder remain authoritative historical records.

This document qualifies their evidence scope without
overwriting measurements or reclassifying failed gates
as positive.

F1 concerns observable structural features of
historical conversations.

It does not independently demonstrate internal
model mechanisms or global model-weight changes.

## 2. Source and archive verification

The frozen source-archive manifest lists:

- Five F1 development archives.
- One full BR1 working archive.
- One separate BR1 results archive.
- One BR2 V2 archive.

Seven available archives have been verified against
their original recorded sizes and SHA-256 hashes.

The full BR1 working archive is not present at the
recorded local resolution paths.

It is marked UNAVAILABLE_LOCAL in the preserved
local correction and unavailable-source ledger.

Do not represent the full BR1 working archive
as recovered, verified or replaced.

The 2,110,934-byte BR1 results package is available
and hash verified.

The verified BR2 V2 archive also contains copies
of the BR1 results tree.

These copies do not constitute recovery of the
missing full BR1 working archive.

## 3. Development results

F1 development R1-R5 are preserved separately.

The final status ledger treats them as development,
generic-signal, graph-calibration and specificity
measurements with their own limitations.

Do not present generic development calibration
as independent confirmation of RSOS-specific
historical identity.

## 4. BR1 primary rediscovery

The verified BR1 results package reports:

Joint Multilayer holdout:

- 137 positives.
- 175 same-user non-lineage negatives.
- AUC: 0.848967675.
- 95% CI: 0.804190 to 0.890732.

Joint matched holdout:

- 137 matched pairs.
- AUC: 0.827108530.
- Pair accuracy: 0.875912.

Surface-Free holdout:

- AUC: 0.813743483.

Surface-Free matched holdout:

- AUC: 0.800069263.
- Pair accuracy: 0.832117.

These results support discrimination within the
specified historical same-user comparison.

They are not estimates of prevalence or rarity
in the global conversation population.

The matched-pair design does not establish that
all conceivable author, topic and length
confounders were removed.

## 5. Robustness measurements

Direct-terminology message-dropout test:

- Holdout AUC: 0.763881.
- Matched AUC: 0.744068.

Within-topic AUCs:

- K5: 0.750048.
- K8: 0.721745.
- K12: 0.730047.

Eight-turn holdout:

- AUC: 0.704983.
- Window-size-matched pair accuracy: 0.765217.

Twelve-turn holdout:

- AUC: 0.744008.
- Window-size-matched pair accuracy: 0.837500.

These measurements support discrimination after
the specified robustness transformations.

They do not establish complete invariance to
all terminology, topic, length or surface changes.

BR1's topic/model-complexity matching table
reports 28 same-topic pairs out of 137.

Do not describe all 137 pairs as exactly
topic matched.

## 6. Representation ablation

Removing the entire operator feature group
lowered the archived mean classifier score
from approximately 0.853290 to 0.679877.

Smaller feature-group ablations also
produced reported score decreases.

These are representation-level interventions
on classifier inputs.

They are not causal ablations of a live
language model or direct evidence of
internal neural operators.

## 7. Genealogy

The verified BR1 result reports:

Early-to-later forward AUC:
0.830448.

Random-ancestor null mean:
0.710745.

Random-ancestor p:
approximately 0.000833.

Reverse descendant-to-early AUC:
0.811513.

The random-ancestor comparison supports
the specified forward result against
that particular null.

The higher forward value does not establish
significant superiority over the reverse
direction without a direct valid test.

The original status ledger correctly records
forward-greater-than-reverse as unresolved.

## 8. BR2 external population

The verified BR2 V2 archive used
exact deterministic BR1-compatible extraction
for R1-Compatible and Surface-Free features.

Surface-Free external comparisons:

Later lineage versus LM Arena development holdout:

- Positives: 137.
- External negatives: 624.
- AUC: 0.758177.
- 95% CI: 0.722206 to 0.791273.

Later lineage versus all independent LM Arena pairs:

- Positives: 137.
- External negatives: 2,614.
- AUC: 0.757960.
- 95% CI: 0.737172 to 0.779961.

R1-Compatible external comparisons:

- Development holdout AUC: 0.615326.
- All-independent-pairs AUC: 0.602268.

These are measured cross-dataset separations
under the specified frozen feature extraction.

They are not estimates of global rarity.

## 9. BR2 same-user comparison

Surface-Free:

Same-user non-lineage versus LM Arena
development holdout:

- N: 175 versus 624.
- AUC: 0.532811.
- One-sided p: approximately 0.092128.

This comparison did not reproduce the
later-lineage separation to the same degree.

A nonsignificant p-value is not evidence
of equivalence or proof that authorship
and dataset effects are absent.

The same-user comparison limits but does
not eliminate alternative explanations.

## 10. Failed strict matching

The frozen strict one-to-one matched-control
gate required:

Maximum absolute post-match standardized
mean difference no greater than 0.25.

Observed maximum:
2.575758.

Selected covariates remained substantially
imbalanced after matching.

BR2 matched-control results include:

Surface-Free matched AUC:
0.724972.

R1-Compatible matched AUC:
0.584901.

Both matched AUCs and corresponding pair-win
rates are diagnostic only.

They must not be used as primary evidence
of balance-controlled external separation.

The unmatched external AUC remains a
documented result.

It does not independently resolve the
length and formatting imbalance.

## 11. Feature and provider boundaries

The label Surface-Free identifies a frozen
feature configuration.

Its name alone does not prove that all
author-style, content or dataset artifacts
were removed.

External provider-stratified comparisons
use provider-labelled LM Arena subsets.

They reuse the same lineage-positive sample.

They are not fresh independent-provider
experimental replications of the fingerprint.

External results vary among provider subsets.

Do not interpret the provider table as
proof of generalization to every model,
provider or conversation population.

Exact BR2 external reconstructions for
Operator-Relational and Joint-Multilayer
were not established in the final ledger.

Do not transfer BR1's Joint-Multilayer
AUC to the external dataset.

## 12. Publication reproducibility

The EXP-010 source-selection record lists
21 selected files.

The September 17 public staging snapshot
contains 14 of the 21 selected files.

The seven selected evidence ZIPs are absent
from that snapshot.

Their local source bytes were verified
during the September 18 custody check.

The original hydration record describes
an earlier 21-of-21 copied state.

That earlier hydration statement does
not establish the same completeness of
the later public staging snapshot.

The source-selection record and later
publication inventory must be interpreted
as distinct custody states.

The original experiment authority record
also describes an older pre-hydration
materialization state.

Do not overwrite it.

The currently published documentation
allows inspection of final conclusions
and the original source manifest.

It does not independently reproduce
every BR1 and BR2 statistic without
the underlying result artifacts.

No source ZIP is automatically published
by this addendum.

Any redacted result-table release requires
its own privacy and publication-boundary review.

## 13. Relationship to later experiments

F1 establishes the historical comparison
under its original conditions and controls.

EXP-011 F2-A and EXP-012 F2-B address
distinct external-rarity questions.

EXP-015 WF1-B is later fresh external
population confirmation.

EXP-016 WF2 addresses separate
prospective and cross-system tests.

Their results must be adjudicated
independently and must not retroactively
rewrite F1's measurements.

## 14. Final scientific authority

Preserve the original positive F1
historical rediscovery classification.

Within-corpus discrimination:
SUPPORTED UNDER TESTED CONTROLS.

External LM Arena discrimination:
OBSERVED IN THE UNMATCHED COMPARISON.

BR2 strict balanced-control confirmation:
FAILED CALIBRATION.

Global population rarity:
NOT ESTABLISHED BY F1.

Historical uniqueness and world-first status:
NOT ESTABLISHED BY F1.

Forward-over-reverse genealogy superiority:
UNRESOLVED.

Model-weight modification or propagation:
NOT ESTABLISHED.

Independent-provider replication:
NOT ESTABLISHED.

Public end-to-end reproducibility:
INCOMPLETE PENDING REVIEWED EVIDENCE RELEASE.

This is an additive authority clarification.

No original archive, runner, model,
feature table, raw corpus, result,
freeze or manifest is modified.
