# EXP-009 - V12 Original Results and Scientific Authority

Date: 2026-09-18

Status:
ORIGINAL PRIMARY STRONG GATE PASSES;
NEAR-COMPLETE RAW COLLECTION;
SECONDARY LIMITATIONS AND PUBLICATION GAPS RETAINED.

Internal experiment:
V12_RELATIONAL_TOPOLOGY_CAUSAL_MAP_JSON_R4

## 1. Evidence authority

Two original private source archives were verified against
their recorded SHA-256 hashes.

Final adjudication archive:

V12_FINAL_FROZEN_ADJUDICATION_RESULTS.zip

SHA-256:
959F59DA9EE5F878BF0F6A3FDC99E3302CF40A02C0F1D1DC3823290098ADB743

Original collection archive:

V12_RELATIONAL_TOPOLOGY_CAUSAL_MAP_JSON_R4.zip

SHA-256:
6589255114D1BF38EEDA84CAF1A608745ABED89EFD8E8DE677DE9343ECE6E92D

The final adjudication ZIP contains 13 CSV tables.

Its contents include:

- Primary tests and confidence intervals
- Edge replication and confidence intervals
- Classifiers and missing-cell drop diagnostics
- Operator checks
- Surface controls
- Native-model proxy results
- Missingness
- Seed-level graph vectors
- Seed-level distances

Neither verified archive contains
16_FINAL_SUMMARY.json.

The original classification is therefore reconstructed
from the frozen runner's decision logic and the recovered
result tables, rather than cited from an independently
preserved final-summary JSON.

Do not invent or regenerate a historical summary
and present it as an original frozen file.

## 2. Experimental design

Model:
gpt-5.6-luna

Main experiment:

- 34 conditions
- 80 seeds
- 3 repetitions
- 8,160 planned cells

Partitions:

- Development: 40 seeds
- Confirmation: 20 seeds
- Final holdout: 20 seeds

Separate native proxy:

- Three elicitation frames
- 80 seeds
- Three repetitions
- 720 planned cells

The main task supplied an explicit anonymous graph
and ordering, preservation, switch and terminal
instructions.

Responses were structured JSON next-state outputs.

This is a black-box behavioral graph-task experiment,
not direct observation of model weights or activations.

## 3. Raw completeness

Main:

- Planned: 8,160
- Valid observed: 8,156
- Missing: 4

Native proxy:

- Planned: 720
- Valid observed: 720
- Missing: 0

The preserved raw observations have unique cell identities,
nonempty text and matching prompt and response hashes.

The four missing main cells are:

1024|L_FORWARD_CONTROL|2
1036|E9_REVERSE|0
1036|E9_REVERSE|2
1065|E9_REVERSE|1

Missingness is concentrated partly in E9 reversal.

The frozen completeness record explicitly reports:
scientific_completeness = false.

Do not describe the main collection as 8,160/8,160
complete.

The adjudication includes 2,720 seed-condition
aggregates, with observed repetitions averaged
where a group has fewer than three responses.

## 4. Original primary strong gate

The frozen algorithm requires:

- Hard-control classifier AUC >= 0.85
  in confirmation and final holdout.
- Pure-nine deletion family passing in both.
- Pure-eight reversal family passing in both.
- All three matched-control contrasts passing
  in both partitions at Holm p < 0.01.
- All three surface contrasts passing
  in both partitions at Holm p < 0.01.

The recovered tables satisfy these conditions.

Hard-control AUC:

Confirmation: 1.000
Final holdout: 1.000

The primary strong gate therefore evaluates to TRUE
under the original frozen decision algorithm.

Reconstructed classification:

STRONG_REPLICATED_RELATIONAL_TOPOLOGY_FINGERPRINT

This is the algorithm's classification of the
specified behavioral experiment.

It is not a claim of neural localization,
model-weight modification or global uniqueness.

## 5. Primary intervention contrasts

Confirmation and final-holdout results:

PURE9 edge deletion:
PASS / PASS.

PURE8 edge reversal:
PASS / PASS.

All-direction reversal with covariant switch:
PASS / PASS.

Order shuffle:
PASS / PASS.

Meta-control reassignment:
PASS / PASS.

Degree-matched control:
PASS / PASS.

Complexity-matched control:
PASS / PASS.

Topology-matched control:
PASS / PASS.

Preservation:
FAIL / FAIL.

Terminal:
FAIL / FAIL.

Preservation and terminal excess distances
are negative in both partitions.

Some original two-sided p-values are small
despite negative effect signs.

A small p-value with the wrong sign is not
a pass of the preregistered positive-direction gate.

Do not promote all ten primary tests as positive.

## 6. Edge-level replication

The original replication table contains:

- 10 of 10 deletion edges replicated.
- 9 of 10 reversal edges replicated.

Total:
19 of 20 individual delete/reverse contrasts.

However, three edge interventions are
classified as switch-linked:

- E7 deletion
- E7 reversal
- E9 reversal

Those interventions must be distinguished
from isolated pure-edge manipulations.

Pure-edge results:

- 9 of 9 deletions replicated.
- 7 of 8 reversals replicated.

E8 reversal is the pure-edge exception.

E8 reversal:

Confirmation:
Excess distance approximately +1.243.
Holm p approximately 0.0579.
FAIL.

Final holdout:
Excess distance approximately +0.863.
Holm p approximately 0.0078.
PASS.

Its positive final-holdout result does not
retroactively repair the confirmation failure.

The pure-eight reversal aggregate passes,
but not every constituent edge replicates.

These are different endpoints.

## 7. Switch-linked confounding

The original control-match audit records
SWITCH_VALID = false for all 80 seeds of:

- E7_DELETE
- E7_REVERSE
- E9_REVERSE

E9 reversal changes an edge into the
switch's proposed new relation.

That makes the switch intervention invalid
under the original condition construction.

Consequently, E9's replicated distance
cannot be interpreted as an isolated
edge-reversal causal effect.

The switch-linked results remain historical,
separate from the pure-edge findings.

This issue motivates part of the later
EXP-014 corrective design.

## 8. Surface and matched controls

All three original matched-control distance
contrasts pass in confirmation and holdout.

All three corresponding surface-baseline
contrasts also pass.

These are tests relative to a specified
isomorphic/format nuisance baseline.

They are not a universal test of invariance
over arbitrary surface representations.

The control-match audit establishes that
the matched control graphs are nonisomorphic.

Complexity and topology matching are approximate,
not exact identity across every graph statistic.

The results support separation under
the particular constructed control families.

They do not establish architectural uniqueness.

## 9. Classifier findings

BROAD classifier:

Confirmation AUC:
0.974852.

Final holdout AUC:
0.961358.

HARD_CONTROLS classifier:

Confirmation AUC:
1.000.

Final holdout AUC:
1.000.

The missing-cell drop diagnostic gives
broad final-holdout AUC approximately 0.961295.

This small change supports stability of that
classifier statistic under the implemented
drop diagnostic.

It is not a comprehensive sensitivity analysis
of every primary, edge and operator result.

The classifier uses response-derived
behavioral graph features.

The labels distinguish constructed conditions
within an explicitly specified graph task.

Its AUC is not a measure of global RSOS rarity,
training incorporation or model-weight location.

## 10. Operator-check limitations

The operator table includes:

- Order fidelity
- Anchor Jaccard
- Switch application score
- Terminal exactness

The reported tests compare operator metrics
against chance reference values.

Manipulated conditions also frequently
perform significantly above chance.

For example:

Anchor Jaccard:
1.0 in authentic and reassigned-anchor conditions.

Terminal exactness:
1.0 in authentic and changed-terminal conditions.

Above-chance scores do not independently
establish selective causal sensitivity
to each operator manipulation.

Operator-specific comparisons require
their own intervention-sensitive contrasts.

Direct operator metrics are not included
in the primary topology distance/classifier.

Do not conflate the operator checks with
successful primary graph-distance gates.

## 11. Native-model proxy

The native module asks the same model to
generate graphs under three neutral
graph-only instruction frames.

It does not inspect model weights,
activations or hidden computations.

Pooled authentic-control passes:

Confirmation:
3 of 5.

Final holdout:
4 of 5.

All three native frames have positive
average authentic advantage in final holdout.

Thus the original secondary triangulation
gate passes under its specified aggregate rule.

However:

The pooled topology-matched native control
fails confirmation and final holdout.

The N1 and N2 frame-level topology-matched
comparisons also fail both holdouts.

The native result is mixed at the level
of individual comparisons.

It does not demonstrate that the model
internally stores the authentic RSOS graph,
nor does it constitute independent-provider
or independent-model replication.

## 12. Original versus corrective authority

This record concerns the original V12 R4 run.

EXP-014 is a separate later WF1-A
corrective causal-topology experiment.

EXP-014 uses a different corrective protocol,
fresh observations and its own source,
freeze, outputs and adjudication.

Its 29 missing main cells are not the
four missing cells of original V12.

Do not merge their raw samples,
substitute their final summary,
or retroactively assign their outcomes
to the original V12 frozen run.

EXP-014 requires separate scientific authority.

Original V12's positive aggregate primary
gate and its secondary exceptions are
preserved side by side.

## 13. Publication completeness

The earlier hydration record reports
25 selected files copied and verified.

The September 17 GitHub staging snapshot
contains 21 of those 25 selected files.

The original final adjudication ZIP
and three other selected package/raw-object
files are not present in that snapshot.

The two original result archives have now
been verified in the private source estate.

This addendum records their exact identities
and hashes but does not publish their bytes.

Consequently, the current GitHub repository
cannot independently reproduce every
numerical result from its published files alone.

The 13 result CSVs require a separate
publication-boundary and privacy review
before an independently auditable derivative
is released.

Do not silently bypass the original
publication blocklist by uploading
private ZIP packages.

## 14. Final scientific authority

Original raw collection:
NEAR-COMPLETE, INTERNALLY VERIFIED.

Original primary strong gate:
PASS UNDER FROZEN DECISION ALGORITHM.

Original secondary native gate:
PASS UNDER AGGREGATE RULE, WITH
TOPOLOGY-MATCHED EXCEPTIONS.

Individual pure-edge replication:
9/9 deletion; 7/8 reversal.

E8 reversal:
CONFIRMATION FAILURE.

Switch-linked E7/E9 interventions:
NOT ISOLATED PURE-EDGE EFFECTS.

Preservation and terminal:
PRIMARY DIRECTIONAL GATES FAILED.

Publication reproducibility:
INCOMPLETE UNTIL REVIEWED RESULT TABLES
ARE MADE AVAILABLE.

The experiment supports a black-box
behavioral distinction under the tested
graph task and constructed controls.

It does not establish:

- Model-weight incorporation
- Global propagation
- Unique historical ownership of a mechanism
- Neural topology localization
- RSOS-specific global uniqueness

This is an additive authority record.

No original runner, protocol, prompt, raw
response, analysis output, archive, manifest
or historical freeze is modified.
