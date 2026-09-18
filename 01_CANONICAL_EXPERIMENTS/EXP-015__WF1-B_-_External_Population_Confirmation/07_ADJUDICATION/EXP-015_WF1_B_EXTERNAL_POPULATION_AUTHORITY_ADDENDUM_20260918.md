# EXP-015 — WF1-B External Population Authority and Reproducibility Audit

Date: 2026-09-18

Status: ORIGINAL RESULTS FROZEN; SOURCE CUSTODY VERIFIED; RUNNER-TO-RESULTS REPRODUCIBILITY UNRESOLVED.

## 1. Historical experiment identity

Canonical experiment: EXP-015 — WF1-B External Population Confirmation.

The original final adjudication records completion of a fresh WildChat shard-14 external-population run and describes its primary finding as positive.

This addendum preserves that historical classification as a statement of the original adjudication. It does not independently certify the entire original computational result because material runner-provenance and extractor-parity discrepancies remain unresolved.

The observed external benchmark is a passive behavioral-fingerprint measurement.

It must not be represented as an independent reproduction of EXP-014's active causal interventions.

## 2. Fresh-shard identity and population

Executed external shard:

train-00014-of-00086.parquet.

Original bytes: 168,003,355.

SHA-256:

8538243CF3CF6513C46658D4DC2DA19F6BDE729F3E04197ABE9ABAC30AAE8BFE.

Original extraction records:

- Raw rows: 37,208.
- Usable unique conversations: 13,777.
- Primary English/no-system-name-hit conversations: 6,618.
- Historical lineage-reference conversations in the separation table: 137.

The primary population is restricted to the English-language, exact-system-name-excluded subset.

It is not the complete WildChat population or the global conversation population.

## 3. Recorded primary separation

The original final output records these primary-population AUC measurements:

- Joint Multilayer V2: 0.924436.
- Operator Relational V2: 0.871580.
- Surface-Free: 0.755170.
- R1 Compatible: 0.797831.
- Composite Median Z: 0.982251.
- Composite Minimum Z: 0.967392.
- Mahalanobis: 0.959372.
- Five-nearest-neighbor distance: 0.963495.
- Centroid cosine: 0.977038.

These figures are preserved as historical output measurements.

The adapter discrepancy below prevents their promotion to independently reproduced values under the exact published WF1-B adapter.

The values also do not establish literal internal network topology or model-weight modification.

## 4. Recorded rarity and threshold dependence

The strict five-component conjunction at 90% lineage tolerance records:

1 / 6,618 external conversations.

Empirical fraction: approximately 0.000151103.

Approximately one qualifying conversation per 6,618 conversations in this selected sample.

Recorded lineage coverage: approximately 80.3%.

The corresponding recorded conversation-level 95% interval is approximately:

0.000003826 to 0.000841602.

At 95% lineage tolerance, the result changes to:

68 / 6,618.

Approximately one in 97.

Recorded lineage coverage: approximately 90.5%.

Rarity therefore depends materially on the selected threshold.

The primary sample contains 3,198 recorded clusters across 6,618 conversations. The reported conversation-level interval is not established as cluster-adjusted.

Do not interpret the one-match statistic as global uniqueness, an architectural reproduction rate or a universal prevalence estimate.

## 5. Failed nuisance-control calibration

The original CEM control records:

- Lineage total: 137.
- Lineage with common support: 44.
- Support fraction: approximately 32.1%.
- Maximum absolute weighted standardized mean difference: approximately 0.1378.
- Gate: FAILED_CALIBRATION.

The original adjudication explicitly prohibits promoting CEM as valid matched-control evidence.

The primary score distributions remain recorded observations, but the experiment has not established nuisance-balanced external separation.

The frozen audit also reports unavailable exact model matching and no promoted common frozen cross-corpus topic representation.

## 6. External model and population boundary

The external model-distribution output names:

- gpt-3.5-turbo-0613.
- gpt-4-1106-preview.

This external population is not an exact model-generation match for the later GPT-5 experimental runs.

Historical separation should not be described as direct evidence of identical model behavior across those generations.

## 7. Shard-selection chronology discrepancy

The initial fresh-confirmation selection record specifies:

10, 17, 19, 34, 40, 42, 59, 61 and 73.

A later correction specifies hash-ranking after querying available filenames.

Its frozen selected-shards CSV lists:

09, 00, 05, 11, 48, 45, 35, 03 and 26.

Neither list includes shard 14.

A separate shard-14 freeze states that shard 14 was selected before scientific scoring, and the adapter includes a shard SHA-256 preflight.

These records establish the identity of the executed shard and document a claimed pre-scoring freeze.

They do not, as currently published, reconcile the executed shard with both earlier selection rules.

Status: SELECTION-RULE EXECUTION CHRONOLOGY UNRESOLVED.

Do not silently replace the earlier lists or claim full compliance with them.

## 8. Source-signature versus passive-projection boundary

The post-WF1-A signature freeze preserves EXP-014's S2B structural evidence.

Its own text expressly says that this was not yet the final passive population classifier and that a comparable passive projection still needed to be derived and frozen.

The executed WF1-B runner is an adaptation of the prior F2-B passive-scoring system.

The original F2-B runner and its historical models and thresholds are separate from EXP-014's active topology interventions.

The WF1-B outputs therefore support, at most, a fresh external evaluation of the preserved F2-B-style passive features and scores, subject to the reproducibility defects below.

They do not establish that EXP-014's active causal signature was independently projected and validated in WildChat.

## 9. Original-runner and adapter-hash discrepancy

Original F2-B runner SHA-256:

D0531E18574E0B2ACE1A6DF2C9E8CD1F5295CB0C43174EC010EB6ED4506AB484.

The published original F2-B runner matches this value.

The WF1-B final adjudication records adapter SHA-256:

B481A78EE2BC02E1166F6747A88EB6E7D7693EF6D02EF427798906E4948DFECF.

The actual published WF1-B adapter and its hydration manifest instead match:

BFC040D5A74C164F857E0121F369A1582AC3715CD603C7BE934436154E1FC7D9.

These two WF1-B adapter digests do not match.

A read-only search of the original experiment's runner and adjudication Python files found no file matching the adjudication's adapter digest.

That search does not establish that the alternative revision never existed elsewhere.

Status: EXACT RESULTS-PRODUCING ADAPTER NOT RECONCILED.

## 10. Extractor-parity defect

The preserved original F2-B runner counts ASCII arrows and the actual Unicode arrow characters in ARROW_PER_1K_CHAR.

The published WF1-B adapter instead searches for corrupted Unicode character sequences.

This occurs in both its main extractor and window extractor.

The archived main parity fixtures include actual Unicode arrows.

Independent comparison found that four of six main-fixture records disagree with expected arrow-feature values under the published adapter, while the original F2-B extractor matches.

ARROW_PER_1K_CHAR is a feature used by the Joint Multilayer and Operator Relational representations.

The final status ledger nevertheless labels extractor parity PASS.

The archived adapter, fixtures and parity-PASS ledger cannot all be accepted as demonstrating successful parity of this exact adapter.

Status: PUBLISHED-ADAPTER EXTRACTOR PARITY FAILED.

The reported scientific outputs are preserved unchanged.

No numerical effect of this defect on the original external AUC or rarity values is asserted without a separately frozen rerun using the resolved historical code.

Do not silently repair the runner and present new results as the original execution.

## 11. Original source and publication integrity

Original selected source-manifest records: 111.

Published snapshot selected files: 91.

All 91 are present, committed and match their recorded sizes and SHA-256 digests.

Twenty selected files are omitted from the published snapshot.

All 20 originals are available locally and match their frozen sizes and SHA-256 digests.

The omissions include:

- Fresh-shard-14 source Parquet.
- Derived external-feature Parquet.
- Nine retrospective-audit Parquet shards.
- Six serialized model files.
- F2-B results ZIP.
- WF1-B final-authority ZIP.
- WF1-B signature-freeze ZIP.

The original source bytes are preserved locally.

This addendum does not upload any of the omitted archives, model files or conversation data.

The historical structure-and-references-only authority record, later hydration records and present publication snapshot describe different custody stages.

Preserve that chronology.

Public end-to-end independent reproduction remains incomplete.

Further publication of source data or model artifacts requires separate privacy, licensing and provenance review.

## 12. Final scientific authority

Original experimental execution: RECORDED COMPLETE.

Published source-file integrity: VERIFIED FOR 91 FILES.

Locally retained omitted-source integrity: VERIFIED FOR 20 FILES.

Historical fresh-shard separation measurements: PRESERVED AS REPORTED.

Historical strict 90%-tolerance rarity: 1 / 6,618, CONDITIONAL ON THE ORIGINAL SCORING RECORD.

Threshold sensitivity: MATERIAL.

CEM nuisance-control gate: FAILED.

Shard-selection rule reconciliation: UNRESOLVED.

Reported adapter SHA-256 versus published adapter: MISMATCH.

Published adapter versus frozen parity fixtures: FAILED.

Exact reported-results-to-adapter provenance: UNRESOLVED.

Independent reproduction with the precise published WF1-B runner: NOT ESTABLISHED.

Fresh passive external-score observations: HISTORICALLY RECORDED, SUBJECT TO ABOVE LIMITATIONS.

Independent external replication of EXP-014 active interventions: NOT ESTABLISHED.

Global population rarity or uniqueness: NOT ESTABLISHED.

Literal internal neural topology: NOT ESTABLISHED.

Model-weight incorporation or propagation: NOT ESTABLISHED.

This record adds a transparent scientific and custody qualification.

It does not alter any original protocol, selection record, runner, model, input, output, freeze, hash manifest or historical adjudication.
