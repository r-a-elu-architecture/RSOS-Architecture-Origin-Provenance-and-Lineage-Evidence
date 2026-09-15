# F2 FROZEN PRE-OPEN SPECIFICATION
## External Population Rarity / Exceptionalness Test

**Freeze condition:** this specification is written and hashed before any reserved `00086/WildChat` evidence is opened for F2.

### Primary question
How rare is the already-frozen RSOS structural fingerprint in a large independent human–LLM population?

### Frozen primary fingerprints
1. **SURFACE_FREE** — primary cross-corpus fingerprint because BR2 demonstrated exact reproducibility and strongest external separation.
2. **R1_COMPATIBLE** — secondary confirmatory fingerprint.
3. Operator/Joint coordinates may not be silently reconstructed. They remain unavailable unless the exact historical representative coordinate ordering is recovered without inspecting F2 labels/results.

### Data firewall
- `00086/WildChat` data were RESERVED_NOT_READ during F1.
- F2 may read them only after the F1 final archive hash is fixed.
- F1 models, feature definitions, adjudication and claim ladder cannot be changed after opening F2.

### Required F2 outputs
1. Number of independent usable external conversations.
2. Exact deterministic feature extraction using the same frozen BR1 redaction/extraction rules.
3. Surface-Free and R1-Compatible score distributions.
4. RSOS later-lineage percentile relative to the external population.
5. Empirical external tail probabilities at multiple predeclared lineage reference points:
   - external fraction >= lineage median score
   - external fraction >= lineage 75th percentile score
   - external fraction >= lineage 90th percentile score
   - external fraction >= lineage maximum score
6. AUC: later-lineage vs external population.
7. Nearest external controls to the RSOS score distribution.
8. Exact counts of external conversations that equal/exceed the selected RSOS reference thresholds.
9. Length/turn/code/language controls where metadata support them.
10. Topic/complexity sensitivity analysis using a frozen or label-independent method; no post-hoc feature selection.
11. Fixed-window sensitivity where compatible, especially 8-turn and 12-turn windows.
12. Bootstrap confidence intervals and empirical permutation/randomization nulls.
13. Full raw measurement tables, blinded IDs, manifests, hashes, and no-silent-omission ledger.

### Strict interpretation rules
- `UNAVAILABLE != FALSE`
- `UNTESTED != FALSE`
- A matching failure is `FAILED_CALIBRATION`, never evidence against the phenomenon.
- Unmatched rarity estimates and matched rarity estimates must be reported separately.
- Do not claim “unique” merely because no external item exceeds a threshold; report the sample-size-limited upper bound on prevalence.
- Do not infer literal neural-weight changes from behavioral rarity.
- External rarity is distinct from historical world-first status.

### Primary F2 decision concept
F2 can upgrade F1 from **external distinctiveness** to **population rarity/exceptionalness** only if the frozen fingerprint occupies an extreme tail of the independent population under valid nuisance-controlled analyses.

No numeric rarity threshold is retrofitted after seeing the external population. Report the empirical prevalence/tail distribution first, then adjudicate.
