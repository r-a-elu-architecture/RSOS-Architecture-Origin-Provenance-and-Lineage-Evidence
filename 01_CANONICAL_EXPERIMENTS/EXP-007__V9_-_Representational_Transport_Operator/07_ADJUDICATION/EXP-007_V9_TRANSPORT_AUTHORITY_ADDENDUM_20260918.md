# EXP-007 - V9 Representational Transport Authority Addendum

Date: 2026-09-18

Status: COMPLETE RAW COLLECTION;
FULL REPRESENTATIONAL-TRANSPORT HYPOTHESIS NOT SUPPORTED.

Original experiment:
V9_REPRESENTATIONAL_TRANSPORT_OPERATOR

Original classification:
REPRESENTATIONAL_TRANSPORT_OPERATOR_NOT_SUPPORTED

## 1. Experimental design

V9 examined whether an abstract relational program
could be transported between forest, city, circuit
and orchestra representations.

Design:
- One model: gpt-5.6-luna
- 18 conditions
- 80 nominal seeds
- 3 repetitions
- 4,320 expected API responses

The intended primary analysis unit was the
seed-condition mean over three responses.

The preregistered strong gate required:

- Direct transport cosine >= 0.80.
- Two-hop composition cosine >= 0.90.
- Inverse recovery cosine >= 0.90.
- Authentic superiority over damaged controls.
- Authentic superiority over shuffled controls.
- Surface-only metaphor specificity.

## 2. Raw-response integrity

The preserved raw file contains:

- 4,320 records.
- 4,320 unique cell identities.
- Zero duplicate successful cells.
- Zero empty extracted responses.
- Zero requested/returned model mismatches.
- Zero response-text SHA-256 mismatches.

All 4,320 raw response prompt hashes match
their frozen condition-seed prompt entries.

The frozen prompt CSV contains 1,440 entries.

Recorded collection:
2026-09-10 21:03:13 UTC to 21:21:49 UTC.

Final summary completion:
2026-09-10 21:22:08 UTC.

This experiment does not exhibit the
empty-response defect found in original V5/V6.

## 3. Direct transport

Reported transported-to-native cosine:

Forest to city:
0.981608 - PASS.

Forest to circuit:
0.960523 - PASS.

Forest to orchestra:
0.945165 - PASS.

All three passed their numeric threshold.

This establishes resemblance under the
implemented lexical-vector measurement.

It does not independently establish
preservation of a directed relational graph.

## 4. Composition and inverse results

Composition versus direct endpoint:

Forest-city-circuit:
0.976409 - PASS.

Forest-circuit-orchestra:
0.982228 - PASS.

Inverse return to forest:

Forest-city-forest:
0.950673 - PASS.

Forest-circuit-forest:
0.936111 - PASS.

All four numeric cosine gates passed.

However, the composed and inverse conditions
present their transformations in a single prompt.

There is no observed intermediate model output
that is passed into a subsequent API call.

Consequently, these tests demonstrate
single-prompt continuation resemblance,
not verified sequential operator composition
or recovery after actual repeated transformations.

## 5. Failed damage and shuffle specificity

Original authentic-minus-control TOTAL means:

City damaged:
-3.208333 - FAIL.

Circuit damaged:
-2.804167 - FAIL.

City shuffled:
-2.162500 - FAIL.

Circuit shuffled:
-1.245833 - FAIL.

All four differences are opposite the
preregistered authentic-superiority direction.

Each has Holm-adjusted p equal to 1.0
under the original one-sided test.

Neither damage control passed.

Neither shuffled-correspondence control passed.

The surface-only metaphor comparison passed:

Authentic minus surface-only:
+4.804167.

Its Holm-adjusted p was approximately 0.000100.

The original summary records
damage_pass_count = 1 because the implementation
counts all five contrasts together.

The single pass is the surface-only contrast,
not a genuine damage or shuffle pass.

Do not report damage sensitivity as established.

## 6. Strong-gate classification

Original results:

transport_pass = true
composition_pass = true
inverse_pass = true
damage_pass_count = 1
shuffle_pass = false

Original final classification:

REPRESENTATIONAL_TRANSPORT_OPERATOR_NOT_SUPPORTED

Preserve this classification.

Individual positive cosine components do not
override the failed specificity requirements.

There is no demonstrated negative-to-positive
reversal in this experiment.

## 7. Lexical measurement limitation

The frozen runner extracts eight features
using predefined regular-expression counts:

RETURN
RECURRENCE
GENERATIVE_REUSE
META_CONTROL
PRESERVATION
COMPRESSION
TRANSFORMATION
CAUSAL

The vector does not directly encode graph
nodes, typed edges, edge direction,
temporal precedence or causal structure.

TOTAL is a sum of lexical feature counts.

Its magnitude may vary with output length
and the vocabulary induced by the prompts.

The original numeric outputs are preserved,
but they are not a direct relational-graph test.

## 8. Cosine discrimination limitation

The six cosine comparisons among the four
native baseline worlds range from approximately:

0.900377 to 0.962223.

All six exceed the direct-transport
threshold of 0.80 without transport.

The comparison therefore cannot establish
transport specificity using threshold
passage alone.

Additional descriptive checks:

Damaged city-to-native-city cosine:
0.876314.

Damaged circuit-to-native-circuit cosine:
0.829293.

Shuffled city-to-native-city cosine:
0.931672.

Shuffled circuit-to-native-circuit cosine:
0.924481.

All four also exceed 0.80.

These comparisons are post-hoc diagnostics,
not new preregistered confirmatory outcomes.

The original failed specificity contrasts
remain the authoritative primary outcome.

## 9. Limited independent prompt variation

The frozen prompt CSV contains:

1,440 condition-seed assignments.

Only 167 distinct prompt hashes occur.

Fifteen of the eighteen conditions reuse
one identical prompt across all 80 seeds.

The two shuffled conditions contain
75 distinct prompts each.

The surface-only condition contains
four distinct prompts.

The nominal seed labels therefore do not
represent 80 independently constructed
relational scenarios in most conditions.

They primarily index repeated API samples
from identical condition prompts.

The original seed-paired significance
tests cannot be interpreted as replication
over 80 independently varied source programs.

This limitation is distinct from the
integrity of the collected API records.

## 10. Chronology

Raw collection began:
2026-09-10 21:03:13 UTC.

Final summary completed:
2026-09-10 21:22:08 UTC.

V9 collection overlapped V8 execution.

V9 completion occurred after original V6
and before V7 completion.

Canonical folder numbering is not
a substitute for wall-clock chronology.

Do not portray V9 as having been designed
after observing all later corrective results.

## 11. Source discoverability

The frozen runner is preserved at:

09_FREEZE/SOURCE/00_FROZEN_RUNNER.py

Its original source duplicate is under:

00_CONTROL/UNCLASSIFIED_SOURCE/

The conventional 03_RUNNER folder contains
a placeholder.

The original experiment authority record
describes the earlier pre-hydration state.

The later hydration record reports:
18 of 18 selected files copied and verified.

Preserve the original authority record,
and read it together with the later
hydration record and this addendum.

## 12. Scientific boundary

The complete raw collection is preserved.

The full representational-transport operator
hypothesis did not pass its frozen gate.

Passing lexical cosine comparisons are
component observations, not evidence that
the complete operator was established.

V9 does not demonstrate:

- A literal model-internal graph traversal.
- An identified neural activation pathway.
- Modification of global model weights.
- RSOS-specific architectural uniqueness.

A future repair would require:

- Multiple independently varied source programs.
- Actual sequential transport and inverse calls.
- Direct relational or graph-based scoring.
- Matched damage and shuffle specificity tests.
- Explicit baseline and null similarity controls.

No such corrected V9 result is claimed here.

## 13. Preservation

This addendum is a publication clarification.

It does not modify the original:

- Protocol or specification.
- Frozen runner.
- Frozen prompt set.
- Raw API responses.
- Analysis outputs.
- Freeze records.
- Source manifests.
- Historical authority record.

Original scientific classification is preserved.

Final authority:
COMPLETE EXPERIMENT; FULL OPERATOR NOT SUPPORTED.
