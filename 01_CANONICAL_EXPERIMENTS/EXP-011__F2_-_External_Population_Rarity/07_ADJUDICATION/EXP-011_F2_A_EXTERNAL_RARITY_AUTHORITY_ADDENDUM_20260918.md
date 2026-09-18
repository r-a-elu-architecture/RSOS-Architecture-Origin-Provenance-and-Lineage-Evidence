# EXP-011 - F2-A External Rarity and Scientific Authority

Date: 2026-09-18

Status:
COMPLETED THREE-SHARD EXTERNAL-POPULATION EXPERIMENT;
SURFACE-FREE SEPARATION OBSERVED;
R1-COMPATIBLE POSITIVE SEPARATION NOT SUPPORTED;
MATCHING CALIBRATION FAILED;
PUBLICATION REPRODUCIBILITY INCOMPLETE.

## 1. Historical identity

Internal experiment:

F2-A_WILDCHAT_3SHARD_EXTERNAL_POPULATION_RARITY

Canonical publication folder:

EXP-011__F2_-_External_Population_Rarity

Original completion recorded:

2026-09-11 21:22:13 UTC.

The frozen runner, specification and final summary
agree on the internal experiment identifier.

This is the original F2-A experiment.

It is not F2-B, the corrected fresh-shard
external-rarity experiment.

Preserve both experiments separately.

## 2. Original objective

Apply frozen early-RSOS-derived classifiers
to previously reserved WildChat conversations.

Primary model:

SURFACE_FREE.

Secondary model:

R1_COMPATIBLE.

The experiment did not retrain classifiers
on the external population.

The frozen runner verifies the F1 final
archive hash before the reserved-data read.

It also checks the portable model scores
against the historical BR1 reference.

The preserved model-equivalence table reports
maximum absolute differences of approximately:

R1-Compatible:
2.22e-16.

Surface-Free:
3.33e-16.

These are implementation-consistency checks.

They do not establish that all relevant
cross-population confounders were controlled.

## 3. Frozen external population

Dataset:

allenai/WildChat-4.8M.

Selected shards:

train-00049-of-00086.parquet
train-00080-of-00086.parquet
train-00085-of-00086.parquet

The three shard identities, sizes and SHA-256
hashes are recorded in the frozen specification
and official-shard manifest.

Historical execution reported matching
official shard hashes.

The September 17 public staging snapshot
does not contain the three shard files.

Their original bytes have not been
independently rehashed in this adjudication.

This experiment does not represent a random,
comprehensive census of all WildChat-4.8M.

Rarity estimates must be confined to its
selected, eligible three-shard population.

## 4. Sample and eligibility

Original recorded counts:

Raw rows encountered:
111,622.

Usable before deduplication:
6,765.

Duplicate blind IDs removed:
0.

Usable unique external conversations:
6,765.

Extraction errors:
0.

Minimum extracted conversation characters:
800.

Minimum conversation turns:
3.

Historical lineage reference:

137 later-lineage conversations.

The eligibility and dataset differences
restrict direct numerical comparison
with F1's LM Arena results.

## 5. Primary external separation

Surface-Free:

Lineage N:
137.

External N:
6,765.

AUC:
0.643281489.

95% confidence interval:
0.624539 to 0.663462.

One-sided Mann-Whitney p:
approximately 4.43e-9.

This is an observed positive score separation
in the tested three-shard population.

R1-Compatible:

AUC:
0.482326919.

95% confidence interval:
0.452077 to 0.512451.

One-sided p:
approximately 0.760966.

The R1-Compatible comparison does not
show the hypothesized positive separation.

Do not describe both frozen classifiers
as successful external confirmations.

## 6. Surface-Free tail exceedances

All thresholds were computed from the
frozen later-lineage reference.

External score at least lineage MEDIAN:

2,341 of 6,765.

Fraction:
0.346046.

Empirical one-in:
approximately 2.89.

At least lineage Q75:

2,000 of 6,765.

Fraction:
0.295639.

Empirical one-in:
approximately 3.38.

At least lineage Q90:

1,803 of 6,765.

Fraction:
0.266519.

Empirical one-in:
approximately 3.75.

At least lineage MAX:

123 of 6,765.

Fraction:
0.018182.

Empirical one-in:
55.

These are classifier-score exceedance rates.

They are not counts of independently
verified reproductions of an RSOS architecture.

They do not measure model-weight inclusion
or global uniqueness.

## 7. Probability-score saturation

The portable runner computes classifier
probabilities using a logistic transformation
of frozen linear scores.

The Surface-Free lineage maximum threshold
is exactly 1.0.

The reported external score quantiles
also reach 1.0.

Many high-scoring external conversations
therefore occupy the probability ceiling.

The maximum-score exceedance rate is
sensitive to this saturation and ties.

It cannot be treated as a uniquely
identifying structural fingerprint.

The frozen runner ranks its TOP250 external
examples by highest classifier probability.

Those are high-score controls.

They are not independently established
nearest neighbors under a graph-distance
or structural-equivalence metric.

## 8. R1-Compatible rarity

External conversations meeting or exceeding
the R1-Compatible lineage median:

3,372 of 6,765.

Fraction:
0.498448.

At least lineage Q90:

2,048 of 6,765.

Fraction:
0.302735.

At least lineage maximum:

479 of 6,765.

Fraction:
0.070806.

Empirical one-in at maximum:
approximately 14.12.

These results do not support a rare
unique signature under R1-Compatible.

No global prevalence estimate follows.

## 9. Nuisance-control calibration

The frozen coarsened-exact matching design used:

- Raw character-count bins.
- Raw turn-count bins.
- Code-presence indicator.

Minimum external observations per supported cell:
5.

Required lineage common-support rate:
0.70.

Maximum permitted absolute weighted SMD:
0.25.

Observed:

Supported lineage conversations:
68 of 137.

Lineage common-support rate:
0.496350.

External conversations in supported cells:
1,882.

Supported cells:
27.

Maximum absolute weighted SMD:
0.040846.

The weighted-balance requirement passed
within the supported subset.

The common-support requirement failed.

Original gate:

FAILED_CALIBRATION.

The conditional Surface-Free pair-win
estimate of approximately 0.589919
applies only to the supported subset.

Its recorded 95% interval is approximately:

0.499311 to 0.679040.

The R1-Compatible conditional estimate
is approximately 0.614457.

Neither conditional estimate repairs
the failed full-lineage support gate.

Do not promote them as fully matched
external-population confirmation.

## 10. Unavailable controls

The original omission ledger marks
the following unavailable in F2-A:

Validated cross-corpus language matching.

Exact model-generation matching.

Common frozen cross-corpus topic matching.

Exact eight-turn and twelve-turn models.

Operator-Relational cross-corpus scoring.

Joint-Multilayer cross-corpus scoring.

The later lineage uses a GPT-5 stratum.

The historical WildChat sample contains
older OpenAI-family model generations.

The reported external sample includes
multiple languages.

No exact model- and language-matched
population result is established here.

These are unexecuted or unavailable analyses,
not negative findings and not positive findings.

## 11. Relation to F1

F1's LM Arena external comparison is
a separate experiment with different
population selection and eligibility.

F2-A uses a new three-shard WildChat sample.

Its Surface-Free AUC is lower than
the reported F1 LM Arena external AUC.

A cross-study numerical difference
does not isolate any single cause.

Population, length, language, model,
topic and eligibility may differ.

Do not pool the two AUCs into a single
estimate without a new valid design.

The historical F1 results and freeze
remain unchanged.

## 12. Relation to F2-B

F2-B is a separate corrected fresh-shard
external-population experiment.

Its six fresh shards are:

07, 22, 41, 51, 52 and 54.

It excludes F2-A's 49, 80 and 85 shards.

F2-B additionally measures:

- Joint-Multilayer V2.
- Operator-Relational V2.
- Deterministic window results.
- Raw margins and multivariate geometry.
- Different composite rarity thresholds.

F2-A does not contain those observations.

F2-B's reported strict five-component
rarity frequency must not be assigned
to F2-A.

The two experiments have different
measures, populations and claim boundaries.

F2-B requires independent adjudication
under EXP-012.

## 13. Custody and publication completeness

The original authoritative selection
contains 51 files.

The earlier hydration status records:

51 selected.
51 copied and verified.
Zero copy errors.

The September 17 publication snapshot
contains 40 of these 51 selected files.

All 40 present selected files match
their recorded sizes and SHA-256 hashes.

Eleven selected files are absent.

The missing files comprise:

- The BR1 reference ZIP.
- The original runner ZIP.
- The external feature Parquet.
- Two paths for the complete F2-A ZIP.
- Three complete-package split parts.
- Three original WildChat Parquet shards.

The absent complete ZIP has recorded SHA-256:

cfe15fc8930fc0121501c062b26bda1e5758f93d45acd124a63f5733c01184b2

That digest is preserved as a manifest value.

This addendum does not claim a new local
byte-for-byte verification of the absent ZIP.

The original experiment authority record
describes an older structural publication
state with source bytes not copied.

The hydration record reflects an
intermediate 51-of-51 verified state.

The later GitHub staging snapshot contains
40-of-51 selected source files.

Preserve all three records chronologically.

Do not overwrite the original authority
or hydration records to hide this difference.

## 14. Privacy and reproducibility

The published snapshot includes summary,
runner, frozen specification, thresholds,
aggregate outputs and custody manifests.

It does not include all original data needed
to reproduce feature extraction end to end.

The original result files use anonymous
or pseudonymous identifiers in some tables.

They are not necessarily fully anonymous.

Do not automatically publish omitted
ZIPs, WildChat shards, raw conversation text
or additional identifier-bearing outputs.

Any further public data release requires
a separate publication-boundary review.

## 15. Final scientific authority

Original experiment execution:
COMPLETED.

Selected three-shard sample:
6,765 ELIGIBLE EXTERNAL CONVERSATIONS.

Surface-Free separation:
OBSERVED IN THE UNMATCHED SAMPLE.

R1-Compatible positive separation:
NOT SUPPORTED.

Full-lineage CEM calibration:
FAILED.

Classifier-score threshold exceedances:
MEASURED, WITH SUBSTANTIAL EXTERNAL OVERLAP.

Structural-equivalence rarity:
NOT ESTABLISHED BY THRESHOLD SCORES ALONE.

Joint, operator and window external tests:
NOT EXECUTED IN F2-A.

Global population rarity:
NOT ESTABLISHED.

Model-weight modification or propagation:
NOT ESTABLISHED.

Public end-to-end reproducibility:
INCOMPLETE.

This is an additive publication adjudication.

No historical protocol, runner, frozen model,
input archive, raw data, output, manifest
or original authority record is modified.
