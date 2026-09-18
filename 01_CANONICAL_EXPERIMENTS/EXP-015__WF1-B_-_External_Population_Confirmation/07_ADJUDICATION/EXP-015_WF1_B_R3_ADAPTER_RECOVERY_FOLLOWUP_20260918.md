# EXP-015 — R3 Adapter Recovery and Parity Follow-up

Date: 2026-09-18

Status: ADAPTER IDENTITY RECOVERED; FROZEN FIXTURE PARITY PASSED; ORIGINAL OUTPUT EXECUTION PROVENANCE NOT FULLY ESTABLISHED.

## 1. Scope

This is a dated, additive follow-up to:

EXP-015_WF1_B_EXTERNAL_POPULATION_AUTHORITY_ADDENDUM_20260918.md

That earlier addendum records the evidence available during its audit. It must remain unchanged as a chronological record.

The present document adds newly recovered evidence. It does not replace the original experimental adjudication or recompute its scientific outputs.

## 2. Recovered R3 identity

EXP-015's original final adjudication identifies the WF1-B R3 adapter by SHA-256:

B481A78EE2BC02E1166F6747A88EB6E7D7693EF6D02EF427798906E4948DFECF.

The original EXP-015 builder is:

BUILD_WF1_B_ADAPTER_R3.py.

Its SHA-256 is:

FB1E01DACCA42CC0EC7A866730F4DCFAECBF6663047B9C0FC0045E2B4BEF15F3.

The builder specifies an R3 output path inside the F2-B frozen passive baseline directory.

The R3 adapter was subsequently located in the frozen foundation package preserved under EXP-016:

02_INPUTS/SOURCE/00_FOUNDATION_FROZEN/DETECTOR_PACKAGE/RUN_WF1_B_FRESH14_ADAPTER_R3.py.

The published EXP-016 file and original locally retained WF2 copy both match the exact R3 digest cited in EXP-015's adjudication.

The previously audited top-level EXP-015 adapter has a different digest:

BFC040D5A74C164F857E0121F369A1582AC3715CD603C7BE934436154E1FC7D9.

It is not byte-identical to R3.

The earlier statement that the adjudication-hashed R3 file could not be located is superseded by this recovery.

## 3. Frozen fixture integrity

Five frozen fixture files were compared between the original EXP-015 baseline and the recovered EXP-016 detector package.

All five pairs matched their recorded SHA-256 hashes:

- PARITY_EXPECTED_V2.csv.
- PARITY_FIXTURES.jsonl.
- WINDOW_PARITY_FIXTURE.json.
- WINDOW8_PARITY_EXPECTED.csv.
- WINDOW12_PARITY_EXPECTED.csv.

The recovered R3 adapter contains the correct Unicode arrow characters.

The older top-level adapter's previously demonstrated arrow-feature defect remains a property of that different archived file.

It must not be attributed to R3 after R3's separate identity and parity have been verified.

## 4. Independent parity-only execution

On 2026-09-18, the original WF2 scoring environment was used with the scorer's dedicated --parity-only mode.

The terminal reported:

Frozen extractor parity: PASS 6

PARITY-ONLY — PROSPECTIVE DATA READ: NO

The scorer and published scorer were verified to have identical SHA-256 hashes before that invocation.

The test did not read prospective observations or generate new experimental measurements.

The reported parity pass establishes fixture compatibility for the recovered R3 implementation under this execution environment.

It does not retrospectively authenticate the code path used in every original EXP-015 result.

## 5. Historical references

The source audit located two original EXP-015 files explicitly referring to R3:

- 04_RUNNER/BUILD_WF1_B_ADAPTER_R3.py.
- 09_ADJUDICATION/WF1_B_FINAL_AUTHORITY_R3/00_FINAL_ADJUDICATION.txt.

The builder establishes the intended creation path.

The final adjudication establishes the revision hash claimed by the historical authority record.

The recovered file matches that claimed hash.

The search did not establish a complete contemporaneous execution trace connecting R3, process invocation, original raw output and every frozen score.

Absence of such a trace in the searched locations is not proof that it never existed elsewhere.

## 6. Corrected authority boundary

RESOLVED:

- Historical R3 builder identified.
- Adjudication-cited R3 bytes recovered.
- R3 SHA-256 identity verified.
- Five fixture pairs verified.
- R3 frozen parity-only execution passed.
- Older and newer adapter revisions distinguished.

NOT YET ESTABLISHED:

- Exact execution provenance connecting R3 to every historical EXP-015 output.
- Independent recomputation of the original AUCs and rarity results using recovered R3.
- Resolution of the original shard-selection-rule chronology.
- A successful EXP-015 nuisance-control calibration.
- Independent replication of EXP-014 active topology in the external population.

The original EXP-015 measurements remain preserved historical results, subject to these boundaries.

This follow-up corrects the missing-R3 and parity interpretation without claiming that the entire EXP-015 result-production chain has been independently reproduced.

## 7. Preservation

This is a documentation-only update.

No original runner, model, protocol, input, output, scoring file, threshold, freeze, manifest, scientific result or previous adjudication is modified.

No unpublished source archive or dataset is released.
