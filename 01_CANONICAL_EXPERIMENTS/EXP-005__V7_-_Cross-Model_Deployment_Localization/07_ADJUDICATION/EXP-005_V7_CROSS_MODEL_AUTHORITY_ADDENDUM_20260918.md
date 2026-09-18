# EXP-005 - V7 Cross-Model Scientific Authority Addendum

Date: 2026-09-18

Status: COMPLETE - PARTIAL CROSS-MODEL BEHAVIORAL REPLICATION

Original experiment:
V7_CROSSMODEL_DEPLOYMENT_LOCALIZATION

Original final classification:
PARTIAL_CROSS_MODEL_DEPLOYMENT_REPLICATION

## 1. Purpose and design

V7 tested whether a five-coordinate behavioral effect
vector reproduced across three model configurations
using stateless, surface-blinded Responses API calls.

Models:
- gpt-5.6-luna
- gpt-5.6-terra
- gpt-5.6-sol

Design:
- 3 models
- 12 conditions
- 60 seeds
- 3 repetitions
- 6,480 planned calls

The primary analysis unit was the seed.

The primary endpoint was cross-model geometric
similarity of the five-coordinate effect vectors.

Each model was required to pass at least four
of five coordinate contrasts.

The preregistered strong gate additionally required
all three pairwise cosine similarities to reach 0.85,
cross-model sign and significance replication,
and positive held-out comparisons.

## 2. Raw-response validity

The preserved raw file contains:

- 6,480 successful records
- 6,480 unique cell keys
- 0 duplicate successful cells
- 0 empty extracted response texts
- 0 requested/returned model mismatches
- 0 missing experimental cells

The raw collection ran from:
2026-09-10 20:36:29 UTC
through:
2026-09-10 21:33:28 UTC.

The final summary records completion at:
2026-09-10 21:34:03 UTC.

V7 does not exhibit the empty-extracted-response
defect identified in original V5 and V6.

## 3. Coordinate results

Luna:
5 of 5 coordinate contrasts passed.

Terra:
5 of 5 coordinate contrasts passed.

Sol:
4 of 5 coordinate contrasts passed.

Across all three models:

Passed:
- Direction
- Meta-control
- Preservation
- Terminal

Not replicated across all three:
- Order

Sol's order difference was approximately -0.083,
with adjusted p approximately 0.660.

The order result must not be reported as
a five-coordinate cross-model replication.

## 4. Primary geometric endpoint

Pairwise cosine similarities:

Luna-Terra:
0.772600 - FAIL

Luna-Sol:
0.935952 - PASS

Terra-Sol:
0.541363 - FAIL

Preregistered pairwise threshold:
0.85

Only one of three model pairs passed.

The strong geometric replication gate failed.

The surviving cross-model result is partial,
not complete replication of the effect vector.

## 5. Held-out results

Held-out authentic-minus-damaged differences:

Luna: +3.388889
Terra: +5.922222
Sol: +3.722222

All three held-out comparisons passed under
the original scoring and test procedure.

These were held-out lexical prompt variants.

They were not independent model providers,
independent API accounts, or an external
population of independent historical conversations.

## 6. Lexical metric limitation

The frozen runner calculates the primary TOTAL
score by adding counts of predefined lexical
features in generated response text.

Its coordinate labels identify experimental
condition contrasts.

They do not mean that the scoring function
directly recovered graph edges, directed
relations or temporal precedence.

The TOTAL score is an unnormalized count.

Response lengths vary across models and conditions.

Therefore, output-length effects and vocabulary
overlap remain alternative contributors to the
observed score differences.

The original statistical results are preserved;
this addendum does not substitute a post-hoc
length-adjusted or graph-based result.

## 7. Recursion specificity limitation

Original aggregate means:

Luna authentic: 13.250
Luna recursion control: 16.806

Terra authentic: 8.683
Terra recursion control: 5.778

Sol authentic: 6.856
Sol recursion control: 9.017

The generic-recursion control exceeded authentic
in two of the three model configurations.

This is descriptive control evidence.

The strong gate did not include a separate
preregistered RSOS-uniqueness criterion.

V7 does not establish RSOS-specific uniqueness.

## 8. Surface variation

Luna authentic mean:
13.250

Luna paraphrase means:
8.694 and 8.606

These changes show that score magnitude is
sensitive to surface variation in at least
one tested model configuration.

The original protocol did not specify
a separate surface-invariance pass gate
within its strong-gate definition.

Do not describe V7 as establishing complete
surface invariance.

## 9. Model and deployment boundary

The recorded API calls used independent
stateless response requests.

Requested and returned model identifiers
match in the preserved raw records.

The frozen runner uses OpenAI client instances
without an independently documented
multi-account or multi-provider experimental arm.

The experiment demonstrates observations
across the three specified model configurations.

It does not independently establish:

- Global model-weight localization
- Modification of deployed model weights
- Propagation caused by the user's interactions
- Independence across API accounts or providers
- RSOS-specific uniqueness

The original specification explicitly recognizes
the weight-localization boundary.

## 10. Execution chronology and genealogy

Original V6 completion:
2026-09-10 21:20:23 UTC.

V7 completion:
2026-09-10 21:34:03 UTC.

V6 Clean completion:
2026-09-10 22:55:18 UTC.

V7 was executed after original V6
and before V6 Clean.

It is a distinct cross-model experiment,
not a later confirmatory replication
of V6 Clean's corrected five-coordinate results.

The conversation contains a declaration that
V7 was complete at approximately 21:24 UTC,
before the final raw response recorded at 21:33 UTC.

That declaration does not independently establish
completion of the final preserved response set
at the earlier conversation timestamp.

The preserved final summary and raw-response
timestamps establish the documented execution end.

This chronology discrepancy does not by itself
establish alteration of a frozen protocol.

## 11. Source discoverability and hydration

The frozen runner is preserved under:
09_FREEZE/SOURCE/00_FROZEN_RUNNER.py

A duplicate original runner is also present under:
00_CONTROL/UNCLASSIFIED_SOURCE/

The conventional 03_RUNNER folder contains
a placeholder.

The original experiment authority record
describes a pre-hydration structural state.

The later hydration record reports:
17 of 17 selected source files copied and verified.

Do not overwrite the original authority record.
Read it together with the later hydration status.

## 12. Final authority

Preserve the original classification:

PARTIAL_CROSS_MODEL_DEPLOYMENT_REPLICATION.

The experiment completed its planned collection.

Four coordinates replicated in direction and
original significance across three models.

The order coordinate did not replicate.

Two of three geometric similarity gates failed.

Held-out lexical comparisons passed.

The measurement is lexical, with unresolved
length and specificity limitations.

No conclusion about global weights or
RSOS-specific uniqueness follows from V7 alone.

All historical protocols, runners, raw responses,
outputs, manifests and freeze records remain
unchanged.

This addendum is a publication clarification,
not a new experiment or replacement freeze.
