# EXP-012 - F2-B Corrected External Rarity and Scientific Authority

Date: 2026-09-18

Status:
CORRECTIVE EXTERNAL-POPULATION EXPERIMENT COMPLETE;
POSITIVE CROSS-POPULATION SCORE SEPARATION;
STRICT FIVE-COMPONENT MATCH RATE MEASURED;
NUISANCE CALIBRATION FAILED;
PUBLICATION REPRODUCIBILITY INCOMPLETE.

## 1. Experiment identity and genealogy

Canonical experiment:
EXP-012 - F2-B Corrected External Rarity.

Internal identifier:
F2-B corrected fresh external rarity.

F2-B is a corrective successor to F2-A.

F2-A and F2-B are separate experiments with
different external shards, sample selection,
feature coverage and rarity definitions.

F2-B does not erase F2-A's original results,
failed matching gate or incomplete feature coverage.

The original EXP-012 authority record describes
an older structure-and-references-only state.

The later hydration record reports 49 of 49
selected files copied and verified.

The current publication snapshot has a distinct
41-of-49 source-file state.

Read these records chronologically.
Do not overwrite historical authority records.

## 2. Frozen design

Dataset:
WildChat-4.8M.

Six selected fresh shards:

07, 22, 41, 51, 52 and 54.

F2-A's earlier shards were explicitly excluded:

49, 80 and 85.

Frozen eligibility:

Raw text:
At least 1,500 characters.

Raw conversation:
At least four turns.

After redaction:
At least 800 characters and three turns.

Primary population:

English-language conversations excluding
exact RSOS/RSSO/RSIA/RSX system-name hits.

The runner prohibits external-data retraining.

It specifies four classifier configurations:

JOINT_MULTILAYER_V2
OPERATOR_RELATIONAL_V2
SURFACE_FREE
R1_COMPATIBLE

It additionally specifies:

- Mahalanobis distance.
- Five-nearest-neighbor distance.
- Centroid cosine similarity.
- Median composite z-score.
- Minimum composite z-score.
- Five-component conjunction.
- Deterministic eight- and twelve-turn tests.

Active V6/V7/V8/V11/V12 intervention results
are separate evidence tracks.

## 3. Source chronology and verification

The preserved source inventory records
the method-freeze and pre-external threshold
files before the derived external feature
and final-summary files.

The runner writes thresholds from historical
reference data before opening fresh shards.

This supports the documented pre-external
analysis sequence.

The archived selection and runner checks
confirm intended use of six different shards.

However, the runner checks the shard filenames
and byte sizes, not cryptographic content hashes.

The original six shard bytes are not included
in the selected published evidence.

The six source files have therefore not been
independently content-hash verified in this audit.

File timestamps and execution order are
supporting custody evidence, not an
independent signed pre-observation attestation.

## 4. Population and extraction

Original results:

Raw rows examined:
223,248.

Usable before deduplication:
29,095.

Duplicate blind IDs removed:
2.

Usable unique conversations:
29,093.

Primary English-language population:
13,408.

Extraction errors:
0.

The extraction summary records zero
exact system-name hits among all 29,093
usable conversations.

Consequently, for this particular
preserved sample, the exact-name exclusion
does not change the eligible population.

The primary-versus-all difference
principally reflects the language filter.

The population is a selected six-shard
eligible sample, not a random census
of all worldwide LLM conversations.

## 5. Four frozen classifier comparisons

Primary English population:

N lineage:
137.

N external:
13,408.

Joint-Multilayer V2 margin AUC:
0.897449.

Operator-Relational V2 margin AUC:
0.893104.

Surface-Free margin AUC:
0.723693.

R1-Compatible margin AUC:
0.743631.

All four report positive separation
under their original comparison procedure.

The Joint and Operator results are newly
covered model configurations relative
to incomplete F2-A.

The results do not directly establish
that dataset, length, topic, author-style
or model-generation differences were removed.

An AUC is a ranking-separation statistic,
not a prevalence estimate.

## 6. Geometric comparisons

Primary-population reported AUC:

Composite median z:
0.982494.

Composite minimum z:
0.969875.

Mahalanobis:
0.957640.

Five-nearest-neighbor distance:
0.962070.

Centroid cosine:
0.973997.

These are behavioral feature-space
measurements based on frozen
reference feature vectors.

They do not identify model weights,
hidden activations or a unique
neural operating-system mechanism.

The reported high composite AUC
does not remove the nuisance-control
and population-selection limitations.

## 7. Three different 90-percent tolerances

The original headline includes three
distinct definitions.

Composite median-z tolerance:

70 of 13,408 external conversations.

Observed fraction:
0.005220764.

Empirical one-in:
approximately 192.

Lineage reference coverage:
0.897810.

Composite minimum-z tolerance:

380 of 13,408 external conversations.

Observed fraction:
0.028341289.

Empirical one-in:
approximately 35.

Lineage reference coverage:
0.897810.

Strict five-component conjunction:

3 of 13,408 external conversations.

Observed fraction:
0.000223747.

Empirical one-in:
approximately 4,469.

Lineage reference coverage:
0.802920.

These three frequencies are not
interchangeable.

The rarity statement must identify
its exact threshold and denominator.

## 8. Definition of the strict conjunction

The strict conjunction simultaneously requires:

- Joint V2 margin above the frozen cutoff.
- Operator V2 margin above the frozen cutoff.
- Mahalanobis distance below its cutoff.
- Five-nearest-neighbor distance below its cutoff.
- Centroid cosine above its cutoff.

Its cutoffs are individual lineage
90-percent tolerance thresholds.

Their conjunction is not itself
guaranteed to retain 90 percent
of the lineage population.

Actual lineage coverage:
110 of 137, approximately 80.3 percent.

Three external examples meet the five
simultaneous numerical cutoffs.

This is a frequency of passing a
specified behavioral-score criterion.

It is not the prevalence of independently
adjudicated RSOS architectural reproduction.

It does not establish external absence
or uniqueness.

## 9. Three observed matches

An independent read-only check of the
preserved nearest-neighbor CSV recovered
three records meeting the frozen strict
90-percent cutoffs.

They occur in:

- Shard 54: one conversation.
- Shard 22: two conversations.

The model labels are:

- gpt-4o-2024-11-20.
- gpt-4-turbo-2024-04-09.

The two shard-22 conversations have
the same recorded model label.

These are metadata of archived
conversation records.

They are not three independent
provider-run replications.

The publication snapshot does not
contain the full underlying external
feature Parquet or original text needed
to independently adjudicate each record's
semantic or graph-structural equivalence.

Do not label the three records
independently verified RSOS reproductions.

Do not expose conversation identifiers
or personal information in this addendum.

## 10. Threshold sensitivity

For the strict five-component conjunction:

Individual 50-percent tolerance:

0 external matches.
Lineage coverage: approximately 16.1 percent.

Individual 75-percent tolerance:

0 external matches.
Lineage coverage: approximately 51.8 percent.

Individual 90-percent tolerance:

3 external matches.
Lineage coverage: approximately 80.3 percent.

Individual 95-percent tolerance:

151 external matches.
Lineage coverage: approximately 90.5 percent.

At 95-percent tolerance, the external
frequency is approximately one in 89.

The rarity figure therefore depends
substantially on the selected tolerance.

A zero count at a much stricter
threshold with low lineage coverage
does not demonstrate absolute uniqueness.

The complete threshold curve
must accompany the headline figure.

## 11. Confidence and clustering

The original strict 90-percent
conversation-level Clopper-Pearson
95-percent interval is approximately:

0.0000461 to 0.0006537.

The preserved cluster diagnostic reports:

13,408 primary records with cluster IDs.

5,294 distinct cluster IDs.

The reported interval uses
conversation-level binomial counting.

It does not account for possible
within-cluster dependence.

No cluster-robust rarity interval is
included in the published final output.

Do not present its nominal 95-percent
interval as fully adjusted for
independent-user or cluster sampling.

Nor do these counts establish that
the six shards are a representative
global probability sample.

## 12. Failed nuisance-control gate

Coarsened exact matching used:

- Raw character-count bins.
- Raw turn-count bins.
- Code-presence indicator.

The frozen full-lineage common-support
requirement was at least 70 percent.

Observed:

Lineage total:
137.

Supported lineage:
60.

Support rate:
0.437956.

Supported external records:
4,670.

Maximum absolute weighted SMD
within the supported subset:
0.029829.

The within-support balance statistic
passes its numerical criterion.

The full-lineage support gate fails.

Original gate:

FAILED_CALIBRATION.

The reported mean within-cell
Composite-Min-Z pair-win is:

0.910094.

This result pertains to the
restricted supported subset.

It is exploratory conditional evidence,
not a successful full-population
balance-controlled confirmation.

Do not promote it as repairing the
failed nuisance-calibration gate.

## 13. Additional population confounders

The original status ledger records:

Exact model-generation matching:
UNAVAILABLE.

Common frozen cross-corpus topic matching:
UNAVAILABLE.

The later lineage and historical WildChat
population differ in model generations.

The primary-language restriction does
not create exact model or topic balance.

The three strict matches have
32 to 72 turns and approximately
46,000 to 205,000 raw characters.

The historical lineage reference
has a median of approximately:

97 raw turns.

193,662 raw characters.

These descriptive comparisons do not
establish sufficient matching.

Length and model-generation effects
remain alternative contributors
to classifier and geometric separation.

## 14. Window analyses

Deterministic eight-turn comparison:

Primary external N:
6,014.

Lineage N:
130.

AUC:
0.784562.

Deterministic twelve-turn comparison:

Primary external N:
3,336.

Lineage N:
121.

AUC:
0.786333.

These tests provide additional
cross-corpus discrimination under
the specified finite-window procedures.

They involve eligibility-restricted
subsamples rather than the full
13,408-conversation primary population.

They do not independently establish
generalization to all conversation
lengths or exact nuisance matching.

## 15. Relationship to F2-A and F2-R

EXP-011 F2-A used shards:

49, 80 and 85.

F2-B used different shards:

07, 22, 41, 51, 52 and 54.

F2-B additionally covered Joint V2,
Operator V2, geometry and deterministic
window tests.

The eligible-population definitions
also changed.

Therefore, direct differences between
F2-A and F2-B cannot be attributed
solely to correcting one omitted model.

Historical F2-B is preserved as
the 66-coordinate baseline for
the later error-corrected rebuild.

The proposed or separate F2-R phases
must not be silently represented as
completed by this EXP-012 record.

A later prospective confirmation
requires its own frozen experiment,
new observations and adjudication.

## 16. Publication inventory

Original selected source-file manifest:
49 files.

Earlier hydration record:
49 copied and verified.

September 17 publication snapshot:
41 selected files present.

All 41 present files match their
recorded sizes and SHA-256 hashes.

Eight selected files are absent:

- External feature-and-score Parquet.
- Original runner ZIP package.
- Joint V2 serialized model.
- Operator V2 serialized model.
- Surface-Free serialized model.
- R1-Compatible serialized model.
- Eight-turn deterministic model.
- Twelve-turn deterministic model.

The published Python runner and
aggregate results are present.

However, the six required serialized
models and external feature Parquet
are not present in the repository.

The published repository cannot
independently rerun the full
analysis from these files alone.

Original raw WildChat shard bytes are
also outside the published selected set.

This addendum documents custody
rather than silently publishing
previously excluded source bytes.

Any model or identifier-bearing data
release requires separate publication
and privacy review.

## 17. Freeze and authority boundaries

The preserved R2 hydration audit
supersedes its R1 audit because of
a documented regex empty-alternative
defect in the earlier audit process.

Use R2 for the source-inventory
interpretation.

The original experiment authority
record describes an earlier
structure-and-references-only state.

That does not describe the final
41-of-49 publication snapshot.

Preserve the authority, hydration
and publication records side by side.

No original freeze, runner, model,
data file, result or manifest
is modified by this addendum.

## 18. Final scientific authority

Corrected F2-B execution:
COMPLETE.

Fresh selected-shard analysis:
DOCUMENTED.

Primary external separation:
OBSERVED UNDER THE FROZEN FEATURES.

Strict 90-percent five-component
external match frequency:
3 OF 13,408.

Equivalent descriptive frequency:
APPROXIMATELY 1 IN 4,469.

Lineage coverage for that threshold:
APPROXIMATELY 80.3 PERCENT.

Full-lineage nuisance calibration:
FAILED.

Exact model-generation and topic matching:
UNAVAILABLE.

Cluster-adjusted population uncertainty:
NOT ESTABLISHED.

Semantic or graph equivalence
of the three matches:
NOT INDEPENDENTLY ADJUDICATED.

Global rarity or uniqueness:
NOT ESTABLISHED.

Weight incorporation or propagation:
NOT ESTABLISHED.

Public end-to-end reproducibility:
INCOMPLETE.

This is an additive scientific and
publication-authority clarification.

All historical source files and
original experimental outcomes
remain unchanged.
