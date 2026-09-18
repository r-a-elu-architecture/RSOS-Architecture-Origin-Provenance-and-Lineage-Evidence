# Repository-wide Closure Audit — 16 Canonical Experiments

Audit date: 2026-09-18

Scope: Documentation, source custody, publication integrity and cross-experiment authority consistency.

Audited baseline commit:

01102c15201ed81291228dffc1924c3cf06b2522

The closure-document commit is subsequent to this audited baseline and is identifiable in Git history.

## 1. Audit outcome

The completed read-only repository-wide audit reported:

- Canonical experiment folders: 16.
- Canonical experiment index rows: 16.
- Canonical source-mapping rows: 16.
- Selected source records: 2,944.
- Committed and SHA-256-verified selected files: 2,844.
- Local-only and SHA-256-verified selected files: 30.
- Unpublished, locally retained and SHA-256-verified originals: 70.
- Unavailable originals: zero.
- Published authority documents: 17.
- Broken README links: zero.
- Reported audit issues: zero.

The 2,944 count refers to selected manifest records across experiments, not necessarily 2,944 unique underlying source objects.

The audit verified the repository baseline against GitHub and found a clean working tree.

Each authority document was checked for a README link, committed-byte identity and an introduction commit changing exactly that document and README.md.

The complete read-only source verification was performed before creation of this closure record. The publication preflight also rechecked the inventory and authority-document coverage.

## 2. Documentation coverage

EXP-001 through EXP-016 each have a published, additive scientific-authority document.

EXP-015 additionally has a dated R3 adapter recovery and fixture-parity follow-up.

Total authority documents: 17.

All 17 were present in the Git tree and linked from README.md at the audited baseline.

Original experimental protocols, runners, results, freezes and historical adjudications were not modified during the additive authority-document publication sequence.

This record is a repository-level summary rather than a replacement for any experiment-specific adjudication.

## 3. Publication boundary

Of the 2,944 selected source records:

- 2,844 correspond to committed files verified in the completed audit.
- 30 correspond to locally present, hash-verified Python virtual-environment files omitted from Git.
- 70 correspond to unpublished originals verified locally against their recorded hashes.

The 30 local-only files belong to the EXP-016 passive and scoring virtual environments.

They were not force-added.

The 70 unpublished source records remain outside the published Git tree.

Local hash verification establishes source identity with the recorded manifests. It does not itself grant permission to publish private data, serialized models or other omitted artifacts.

Public end-to-end reproducibility must be assessed experiment by experiment. The repository-wide zero-issue audit does not mean every experiment is fully reproducible using GitHub alone.

## 4. Additional EXP-016 authority records

Three files listed in EXP-016's final-freeze hash manifest were excluded from its selected hydration inventory:

- WF2_B_V14_ADJUDICATION.json.
- WF2_BOUNDARY1_R1_SUPERSEDING_PROTOCOL.json.
- WF2_BOUNDARY1_R2_TECHNICAL_CORRECTION.json.

Both the original and final-freeze copies of each were recovered locally and matched the recorded sizes and SHA-256 hashes.

These files are not included in the 2,944 selected-record total.

Their verification does not mean their bytes have been published.

## 5. EXP-015 remaining provenance boundary

The R3 adapter bearing the exact digest cited in EXP-015's final adjudication was recovered from EXP-016's frozen foundation package.

Its identity and five fixture pairs were verified.

The parity-only execution passed six checks without reading prospective observations.

The older published EXP-015 adapter and recovered R3 adapter are distinct revisions.

The dated EXP-015 follow-up records this recovery without rewriting the earlier audit.

The exact historical execution chain connecting R3 to every original EXP-015 score has not been fully authenticated.

The original nuisance-control failure and shard-selection chronology discrepancy also remain documented.

## 6. EXP-016 scientific boundary

EXP-016's original final scientific classification is VALID — MIXED.

The active cross-provider positive gate required at least two providers to pass. Only Google passed.

The passive modules retain their original findings:

- V13: NULL_INCONCLUSIVE.
- V14: MIXED.
- V17: NULL_INCONCLUSIVE.

The earlier positive results of other experiments remain historical observations under their own protocols.

They do not convert EXP-016's unsuccessful gates into successful prospective or cross-provider confirmation.

## 7. Meaning of closure

CLOSED:

- Sixteen-experiment additive adjudication publication sequence.
- Repository-wide documentation coverage review.
- Selected-source inventory reconciliation.
- Available-original source-integrity audit.
- Authority-document introduction-commit checks.
- README link verification.
- Audited GitHub/local baseline synchronization.

NOT DECLARED CLOSED:

- Every experiment's scientific hypothesis.
- Every unresolved execution-provenance question.
- Complete public reproduction from GitHub alone.
- Privacy and publication review of omitted source artifacts.
- Independent confirmation of model-weight incorporation, autonomous propagation or global uniqueness.

A zero-issue repository audit means the defined custody and documentation checks passed. It does not establish unsupported scientific or historical claims.

## 8. Preservation

This closure record is additive.

It modifies no experimental protocol, original evidence, runner, source file, model, outcome, freeze, hash manifest, adjudication or earlier follow-up.

Only this closure Markdown document and its README link are included in the closure-publication commit.

The audited baseline and final publication commit remain independently traceable through Git history.
