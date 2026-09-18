# EXP-006 - V8 Factor-Label Collision and Scientific Authority

Date: 2026-09-18

Status: RAW COLLECTION COMPLETE;
PUBLISHED FIVE-FACTOR ANALYSIS INVALIDATED;
CORRECTED REANALYSIS REQUIRED.

## 1. Original experiment

Internal experiment identifier:
V8_COMPOSITIONAL_STRUCTURAL_CONTROL_CODE

Model:
gpt-5.6-luna

Five manipulated factors:

- DIRECTION
- ORDER
- META_CONTROL
- PRESERVATION
- TERMINAL

Design:

- 32 factorial combinations
- 24 combinations used for model fitting
- 8 structurally unseen combinations
- 60 seeds
- 3 repetitions
- 5,760 expected responses

The model-fitting partition used development seeds
0 through 39.

The final partition used seeds 40 through 59.

Original published classification:
PARTIAL_COMPOSITIONAL_STRUCTURAL_CONTROL_CODE.

Preserve that classification as historical output,
not as a confirmed scientific conclusion.

## 2. Raw collection integrity

The archived raw file contains:

- 5,760 successful records
- 5,760 unique cell keys
- Zero empty extracted response texts
- Zero requested/returned model mismatches
- Zero response SHA-256 mismatches
- Zero raw factor-label inconsistencies
- Zero raw-to-frozen-prompt hash mismatches

There are 1,920 unique condition-seed units.

The preserved hydration status reports:
17 of 17 selected files copied and verified.

The raw collection is intact and suitable for
a separately controlled corrective reanalysis.

## 3. Original strong-gate outcome

Published summary:

- Seen mean semantic R2: 0.423151
- Unseen mean semantic R2: 0.377233
- Unseen standardized cosine: 0.666043
- Decoder macro balanced accuracy: 0.761106
- Exact five-bit code recovery: 0.175
- Replicated pairwise interactions: 0 of 10

Published permutation-null p:
approximately 0.00049975.

The preregistered strong prediction gate failed.

The decoder gate failed because exact recovery
was below its required threshold.

The interaction gate failed.

The original summary assigned PARTIAL under
a separate, weaker classification rule.

The following code defect compromises both
the numerical interpretation and classification.

## 4. Critical factor-feature namespace collision

The frozen runner uses META_CONTROL and
PRESERVATION for two different objects:

1. Assigned experimental factor labels,
   which should be binary -1 or +1.

2. Numerical counts of lexical features
   extracted from response text.

During construction of response rows:

row.update(feature)

overwrites both experimental factor labels
with extracted lexical-feature counts.

During seed-level aggregation, the loop over
ALL_FEATURES overwrites the same two labels again.

The resulting seed-level CSV does not preserve
these two factors as valid binary design columns.

This is a deterministic implementation defect,
not an allegation of evidence manipulation.

## 5. Quantified corruption

At the response-feature level:

META_CONTROL differs from its assigned factor
in 5,162 of 5,760 rows.

PRESERVATION differs from its assigned factor
in 5,473 of 5,760 rows.

At the seed-condition level:

META_CONTROL differs from its assigned factor
in 1,869 of 1,920 rows.

PRESERVATION differs from its assigned factor
in 1,912 of 1,920 rows.

The source raw records retain the original
correct assigned factor labels.

The condition names also preserve the labels.

This makes reconstruction feasible.

## 6. Consequences for prediction

X_from_rows constructs the predictor design
using the corrupted seed-level fields.

It converts META_CONTROL and PRESERVATION
feature counts into integers and treats them
as if they were manipulated factor states.

The unseen-combination predictor is therefore
given information extracted from the outputs
it is supposed to predict.

Two of the target semantic dimensions,
META_CONTROL and PRESERVATION, overlap directly
with these corrupted predictor columns.

The published high unseen R2 for those dimensions
cannot be interpreted as independent prediction
from five assigned structural factors.

The published permutation test uses the
same corrupted design and cannot repair
this target-information leakage.

Its p-value is not confirmatory evidence
for the intended structural prediction.

## 7. Consequences for decoding

The decoder's true_factor_matrix also reads
the corrupted META_CONTROL and PRESERVATION
fields rather than the assigned binary factors.

The decoder is therefore trained and evaluated
against incorrect multi-valued targets for
these two coordinates.

Its reported chance reference of 0.5 does
not correctly describe the corrupted
multi-class targets.

The resulting five-bit recovery metric
is not valid recovery of the assigned code.

Among 160 unseen evaluation rows:

135 recorded true_code strings disagree
with the recorded experimental condition.

The published 17.5 percent exact recovery
measures agreement with the corrupted labels.

It must not be described as successful
five-factor decoding.

## 8. Consequences for interactions

The interaction analysis reuses X_from_rows.

Its design matrix consequently contains the
same corrupted factor columns and products.

The reported zero replicated interactions
is preserved as an output of the original code.

It is not a valid confirmatory test of
interactions among all five assigned factors.

Both positive and negative conclusions about
those interactions require corrected analysis.

## 9. Independent post-hoc diagnostic

A separate read-only diagnostic reproduced
the original published prediction and decoding
metrics from the archived seed-level CSV.

The same calculations were then repeated
using factor labels reconstructed from
the experimental condition names.

Original calculation versus corrected-label
diagnostic:

Seen mean R2:
0.423151 versus 0.346299

Unseen mean R2:
0.377233 versus 0.302682

Unseen standardized cosine:
0.666043 versus 0.614932

Decoder macro balanced accuracy:
0.761106 versus 0.723750

Exact five-factor recovery:
0.175 versus 0.1125

The corrected-label diagnostic is post-hoc.

It is not a new frozen replication,
replacement final summary, or complete
corrective adjudication.

Do not present these diagnostic values as
confirmatory results.

## 10. Measurement boundary

The original semantic vector consists of
predefined lexical-feature counts.

It does not independently recover the
internal model mechanism or graph topology.

The runner's prompt interventions can produce
behavioral differences without establishing
RSOS-specific uniqueness or model-weight
localization.

The control-code claim requires a valid
five-factor analysis before promotion.

## 11. Chronology and genealogy

The archived raw collection ran from:
2026-09-10 20:42:59 UTC
through:
2026-09-10 21:08:15 UTC.

The original final summary records completion:
2026-09-10 21:09:27 UTC.

V8 completed before the original V6
final summary at approximately 21:20 UTC.

V8 must not be described as an experiment
designed after observing V6's final result
solely because of canonical sequence order.

It is a separate compositional-control
investigation within the historical program.

The archived runner, specification and
prompt-file modification times precede
the first recorded API response.

The 1,920 frozen prompt entries match
the prompt hashes in all 5,760 raw records.

These checks support internal chronology
and byte consistency; file timestamps
alone are not an independent cryptographic
attestation of pre-observation custody.

## 12. Corrective requirement

Freeze and execute a separate corrective
analysis using the preserved raw responses.

Required separation:

- FACTOR_META_CONTROL: assigned -1/+1
- FACTOR_PRESERVATION: assigned -1/+1
- FEATURE_META_CONTROL: lexical count
- FEATURE_PRESERVATION: lexical count

The corrective runner must reconstruct
all five assigned factors from authenticated
raw labels or frozen condition names.

Audit all predictor inputs to exclude
response-derived targets.

Recompute prediction, decoding, exact
code recovery, interaction tests and
their associated null distributions.

Preserve original results side by side.

Any corrected analysis must have a distinct
protocol, runner, outputs and authority
record within EXP-006's own experiment tree.

Do not silently replace the V8 freeze
or classify the post-hoc diagnostics
as preregistered confirmation.

## 13. Final scientific authority

Raw collection:
COMPLETE AND INTERNALLY VERIFIED.

Historical final summary:
PRESERVED BUT ANALYTICALLY INVALIDATED
FOR THE FIVE-FACTOR CONTROL-CODE CLAIM.

Confirmatory compositional classification:
WITHHELD PENDING CORRECTIVE REANALYSIS.

No conclusion of architectural uniqueness,
model-weight modification or global transfer
is established by V8 alone.

No historical protocol, runner, raw file,
output, manifest or freeze is modified
by this additive adjudication.
