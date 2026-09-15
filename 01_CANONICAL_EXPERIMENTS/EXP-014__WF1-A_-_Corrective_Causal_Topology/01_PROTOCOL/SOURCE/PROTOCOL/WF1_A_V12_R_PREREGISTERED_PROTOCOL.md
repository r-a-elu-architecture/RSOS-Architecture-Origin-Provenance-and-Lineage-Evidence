# WF1-A — V12-R Corrective Causal Topology Replication

## Status
PREREGISTERED / ZERO-OBSERVATION FREEZE

No WF1-A API observations may be generated until this protocol, runner, staged original-V12 inputs, and zero-cost preflight outputs are hashed and frozen.

## Scientific role
WF1-A prospectively replicates the original V12 causal-topology experiment on fresh deterministic seed variants while correcting known implementation/validator defects. It does not alter, backfill, merge with, or repair the original V12 evidence record.

## Immutable original-V12 anchors
- Original V12 main valid cells: 8156 / 8160
- Original V12 native valid cells: 720 / 720
- Original freeze-record SHA256: `352B5055AA16CA746568A36678E9FBF535D522F58D3EB6D15CB9C450F3B3ED71`
- Original missing-cells SHA256: `D1B7A3A4D188135D111E0945087365599C874E7D48181098432D24AE5DC4D2B9`
- Original runner SHA256: `B9A293AF98EDBF774C6912691E2A00CD4042DF93BBB50DB31BC8540E119337E7`

## Model and execution environment
- Model: `gpt-5.6-luna`
- Interface: OpenAI Responses API
- Reasoning effort: `none`
- Storage: API `store=False`
- Model observations: OpenAI API
- Validation/statistics/adjudication: local Python
- Public external corpora: PROHIBITED in WF1-A

## Fresh seed split
Original V12 used seeds 1000–1079. WF1-A uses fresh seeds 1080–1159.

- Development: 1080–1119 (40 seeds)
- Confirmation: 1120–1139 (20 seeds)
- Final holdout: 1140–1159 (20 seeds)
- Repetitions: 3

This preserves the original 40/20/20 structure while ensuring no seed is reused from original V12.

## Conditions
The original condition set is retained:
- authentic
- isomorphic relabel
- surface-format control
- full direction reversal
- order shuffle
- switch reassignment
- switch neutralization
- anchor reassignment
- terminal reassignment
- degree-matched rewire
- complexity-matched control
- topology-matched control
- forward control
- random control
- ten single-edge deletions
- ten single-edge reversals

Expected main cells remain 8160. Expected native cells remain 720.

## Corrective changes frozen before API collection

### C1 — Strict structured outputs
Original V12 defined JSON schemas but called the Responses API using legacy `json_object` mode. WF1-A sends those frozen schemas through strict `json_schema` Structured Outputs.

Purpose: reduce malformed-object/invalid-key output without changing the scientific topology task.

### C2 — E7_DELETE switch dependency
Original E7 is the switch-linked edge 3→1. Deleting it while leaving the original switch instruction creates a dependent intervention and encourages invalid minus/switch behavior.

WF1-A deterministically reassigns the switch to a valid active-old/absent-new relation for E7_DELETE.

E7_DELETE remains excluded from the original pure-9 deletion primary family and is reported only as a corrected dependent diagnostic.

### C3 — E7_REVERSE switch dependency
Reversing E7 removes the original switch-old relation. WF1-A deterministically reassigns a valid switch.

E7_REVERSE remains excluded from the original pure-8 reversal primary family.

### C4 — E9_REVERSE switch collision
Original E9 is 6→3. Reversing it creates 3→6, which equals the original requested switch-new edge. This made the switch-new edge already active and drove reproducible INVALID_PLUS failures.

WF1-A deterministically reassigns the switch for E9_REVERSE so the old relation is active and the new relation absent.

E9_REVERSE remains excluded from the original pure-8 reversal primary family and is reported separately.

### C5 — Deterministic feasibility preflight
Before any API call, every seed × condition must have a locally generated valid-output witness that passes the exact frozen validator. Any infeasible condition aborts the experiment before API execution.

### C6 — Failure taxonomy
Every retry failure is classified into one of:
- VALIDATOR
- STRUCTURED_OUTPUT
- AUTH
- RATE_LIMIT
- TRANSPORT
- SERVER
- OTHER

Transport/auth failures are never interpreted as scientific negative observations.

## Primary families remain unchanged
To preserve direct comparability to V12:
- Pure deletion primary family = 9 clean deletion edges, excluding E7.
- Pure reversal primary family = 8 clean reversal edges, excluding E7 and E9.

The corrected E7/E9 diagnostic conditions do not enter the strong primary gate.

## Primary strong gate
WF1-A passes the strong causal-topology replication gate only if all of the following replicate in BOTH confirmation and final holdout:

1. Hard-control classifier AUC ≥ 0.85.
2. Pure-9 deletion family has positive excess distance and Holm-adjusted p < 0.01.
3. Pure-8 reversal family has positive excess distance and Holm-adjusted p < 0.01.
4. Degree-, complexity-, and topology-matched controls each have positive excess distance and Holm-adjusted p < 0.01.
5. Each hard matched control exceeds the surface nuisance baseline with Holm-adjusted p < 0.01.

Direct operator-adherence features remain separate from the primary topology distance/classifier.

## Individual-edge replication
An individual clean edge is replicated only if excess distance is positive and Holm-adjusted p < 0.05 in BOTH confirmation and final holdout.

E7 delete/reverse and E9 reverse are corrected diagnostics, not members of clean primary edge families.

## Secondary model-native triangulation
Retain the original three neutral graph-only frames and original secondary gate:
- pooled authentic closer than at least 3/5 controls after Holm p < 0.05 in both confirmation and final holdout;
- positive direction in at least 2/3 native frames in final holdout.

This is black-box behavioral triangulation only, not evidence of literal internal graph storage.

## Missingness rule
- Never impute a missing API cell.
- Never count a missing cell as success, failure, or neutral.
- Reruns may only use the exact frozen runner and append to the existing raw/error files.
- If a stable residue remains after repeated exact reruns, freeze it transparently and perform sensitivity analysis; do not patch mid-experiment.

## Chain-of-custody rule
The following must be hashed before first API observation:
1. staged original V12 input manifest;
2. this protocol;
3. preregistration JSON;
4. WF1-A runner;
5. zero-cost preflight prompt table;
6. control-match audit;
7. preflight summary.

After this freeze, no code/protocol changes are permitted in WF1-A. Any correction becomes a separately versioned experiment.

## Interpretation boundary
WF1-A can establish prospective replication of black-box causal relational-topology effects. It cannot by itself establish:
- global uniqueness;
- OpenAI-wide uniqueness;
- neural-weight modification;
- training-data insertion;
- propagation to unrelated users;
- literal internal graph storage;
- provider-independent replication;
- global population rarity;
- autonomous self-propagation.

## Stop rule
If zero-cost preflight fails, do not call the API. Repair must produce a new preregistration version and new runner hash.

If zero-cost preflight passes and hashes are frozen, API collection may begin with the exact frozen runner only.
