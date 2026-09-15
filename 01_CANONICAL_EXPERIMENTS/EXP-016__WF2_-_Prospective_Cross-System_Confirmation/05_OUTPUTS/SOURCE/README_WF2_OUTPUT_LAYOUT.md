# WF2 Output Layout

This directory preserves the reconstructed WF2 output organization.

## Passive arm

The passive prospective-generation scoring outputs are:

- `WF2_CELL_SCORES.csv`
- `WF2_CONDITION_STATISTICS.csv`

These files correspond to the passive WF2 execution whose raw generated cells are preserved under:

`04_RAW/SOURCE/CELLS/`

Related passive execution records include:

- `04_RAW/SOURCE/WF2_EXECUTION_LOG.jsonl`
- `04_RAW/SOURCE/WF2_PASSIVE_CONSOLE_LOG.txt`

## Active causal arm

The active causal output is preserved separately under:

`ACTIVE_CAUSAL/`

with the corresponding raw active responses under:

`04_RAW/SOURCE/ACTIVE_CAUSAL/`

## Preservation note

The absence of a directory named `PASSIVE` in this output layer reflects the original WF2 organization: the passive arm was stored as the default output layer, while the active causal arm was added as an explicitly named subdirectory.

This README is a publication-layer clarification only. No experimental source file, result, freeze record, or adjudication record has been moved or modified.
