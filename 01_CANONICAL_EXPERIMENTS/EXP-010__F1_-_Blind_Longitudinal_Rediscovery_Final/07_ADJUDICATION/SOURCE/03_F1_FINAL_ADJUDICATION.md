# F1 FINAL — FREEZE AND ADJUDICATION

## Final F1 classification

**F1 — STRONG BLIND HISTORICAL STRUCTURAL REDISCOVERY WITH INDEPENDENT LM-ARENA POPULATION SEPARATION**

F1 supports a reproducible RSOS-lineage structural fingerprint in the historical corpus. The signal was frozen from early RSOS and recovered later RSOS/RSSO/RSIA/RSX conversations under blinding, survived direct-terminology message dropout, topic/domain controls, local-window length hardening, same-user non-lineage controls, and genealogy testing.

BR2 then applied exact deterministic BR1-compatible extraction to independent LM Arena conversations and reproduced external separation. The strongest exact cross-corpus result is the frozen Surface-Free model:

- later lineage vs LM Arena development holdout: AUC 0.758177, 95% CI 0.722206–0.791273, p=1.36379e-21
- later lineage vs all independent LM Arena pairs: AUC 0.757960, 95% CI 0.737172–0.779961, p=1.0585e-24
- later lineage vs same-user non-lineage: AUC 0.813743
- same-user non-lineage vs LM Arena: AUC 0.532811, p=0.0921282

This pattern materially weakens the explanation that the measured effect is merely generic authorship/style.

## Important limitation

The BR2 strict one-to-one matched-control block FAILED its predeclared balance gate:
max |SMD| after matching = 2.5758, whereas the gate was <=0.25.

Therefore BR2 matched AUCs/pair-win rates are diagnostic only and are NOT used to support the primary F1 conclusion.

## Claim boundary frozen at F1

### Supported
1. A reproducible historical structural fingerprint-like signal exists in the RSOS lineage.
2. Early RSOS structure blindly rediscovers later RSOS/RSSO/RSIA/RSX above same-user controls.
3. The effect survives substantial removal/control of terminology, topic, length, and several nuisance factors.
4. The Surface-Free fingerprint separates later lineage from an independent LM Arena conversation population.
5. The external separation is not reproduced to the same degree by ordinary same-user non-RSOS conversations.

### Not established by F1
1. Global population rarity or uniqueness.
2. World-first/unprecedented status across all human–LLM interaction history.
3. Literal modification of global model neural weights.
4. Propagation of RSOS into unrelated users/models.
5. Prospective prediction on future frozen data.
6. Independent third-party/provider replication.

These are not negative findings. They remain UNTESTED / UNRESOLVED / UNESTABLISHED as listed in the status ledger.

## F1 preservation rule

F2/F3 results may add evidence but MUST NOT retroactively rewrite F1 measurements. Any later failure or success is recorded as a new layer in the evidence chain.
