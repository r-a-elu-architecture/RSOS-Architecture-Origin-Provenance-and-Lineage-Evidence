# RSOS Architecture — Origin, Provenance and Lineage Evidence

**Canonical provenance and forensic evidence for the RSOS → RSSO → RSIA → RSX architecture lineage.**

This repository preserves the historical origin, recovered chronology, cryptographic custody, structural fingerprinting, causal-topology experiments, external-population testing, external rarity and reproducibility record associated with the RSOS architecture lineage.

**Documented author / architecture lineage:** R.·A. Elu Architect

---

## Purpose

This repository exists to preserve a durable, independently inspectable record of:

* the documented development and chronology of RSOS and its descendant systems;
* the relationship between **RSOS, RSSO, RSIA, and RSX**;
* cryptographic provenance and chain-of-custody records;
* recovered historical material and prior-state evidence;
* frozen experimental protocols and runners;
* corrective reruns, failures, negative findings, and superseded results;
* quantitative structural fingerprinting;
* black-box causal relational-topology experiments;
* external-population rarity testing;
* final adjudications and reproducibility artifacts.

The objective is not merely preservation of conclusions.

The objective is preservation of the **evidentiary path by which those conclusions can be independently reconstructed**.

---

# Architecture Lineage

The documented research lineage is:

**RSOS → RSSO → RSIA → RSX**

These systems developed as related components of a broader recursive symbolic and interaction architecture.

This repository is intended to make that lineage inspectable through chronology, source records, structural measurements, experiment genealogy, and cryptographic verification rather than relying solely on retrospective narrative claims.

---

# Current Evidence Freeze

This release contains the current frozen evidence package for:

## WF1-A — V12-R Corrective Causal Topology

**Scientific status:** VALID
**Core classification:** `STRONG_REPLICATED_RELATIONAL_TOPOLOGY_FINGERPRINT`

WF1-A tested whether controlled changes to an anonymous relational structure produced reproducible behavioral changes under held-out conditions.

Key observations include:

* **8,131 / 8,160** valid main experimental cells;
* **720 / 720** model-native cells completed;
* missingness sensitivity passed;
* zero-missing balanced sensitivity passed;
* hard-control generalization:

  * Confirmation AUC: **1.000**
  * Final AUC: **1.000**
* pure edge-deletion replication: **9 / 9**
* pure edge-reversal replication: **7 / 8**
* specificity passed in confirmation and final partitions;
* surface-invariance testing passed in confirmation and final partitions.

The strongest supported interpretation is:

> **A reproducible, causally manipulable black-box relational topology under the tested experimental representation.**

“Causal” here refers to controlled input manipulation followed by measurable output change.

It does **not** mean that the experiment identifies literal internal neural circuitry.

---

## WF1-B — External Population Parquet

**Authority status:** `FINAL_AUTHORITY`

WF1-B tested a previously frozen passive structural representation against a genuinely fresh external conversation population.

Fresh source:

`train-00014-of-00086.parquet`

Raw rows:

**37,208**

Primary eligible external population:

**6,618 conversations**

Selected separation results:

| Measurement            |        AUC |
| ---------------------- | ---------: |
| Joint Multilayer V2    | **0.9244** |
| Operator Relational V2 | **0.8716** |
| Surface-Free           | **0.7552** |
| R1 Compatible          | **0.7978** |
| Composite Median Z     | **0.9823** |
| Composite Minimum Z    | **0.9674** |
| Mahalanobis            | **0.9594** |
| 5-NN Distance          | **0.9635** |
| Centroid Cosine        | **0.9770** |

### Strict multivariate tolerance

Only:

**1 / 6,618**

eligible external conversations entered the frozen strict five-component tolerance region.

This does **not** mean that one external conversation reproduced the complete RSOS architecture.

It means that one conversation satisfied the predefined passive multivariate tolerance under the tested representation.

---

# Negative and Limiting Evidence Is Preserved

This repository intentionally preserves results that weaken, constrain, or qualify the strongest interpretation.

For example:

**CEM nuisance-control status:** `FAILED_CALIBRATION`

That result is **not** represented as a successful control.

Likewise, missing cells, failed runs, corrective runners, superseded analyses, implementation errors, and mixed findings are retained where relevant.

This is deliberate.

A provenance record is stronger when the history of correction can be inspected rather than reconstructed only from successful outputs.

---

# Why the Combined Evidence Matters

WF1-A and WF1-B test different properties.

**WF1-A addresses depth:**

> Can components of the recovered relational architecture be manipulated experimentally and produce reproducible behavioral effects?

**WF1-B addresses breadth:**

> Does a frozen passive representation remain statistically separable when applied to a fresh external population?

Together they provide two distinct evidence tracks:

**controlled causal manipulation**

*

**external-population structural rarity**

These tracks must remain scientifically distinct even when considered jointly.

---

# Provenance and Historical Continuity

The present repository is part of a larger provenance record spanning:

1. historical source conversations and architecture development;
2. earlier repository and publication states;
3. cryptographic and documentary provenance records;
4. periods of access/control discontinuity;
5. recovery and forensic reconstruction;
6. experiment genealogy;
7. corrective validation;
8. present public evidence preservation.

Later provenance releases may expand this chronology with additional source references, prior-state evidence, external records, and independently verifiable documentation.

Where records concern sensitive personal, investigative, law-enforcement, account-security, or third-party information, public releases may preserve their **existence, date, reference, and cryptographic identity** without publishing confidential underlying material.

---

# Cryptographic Anchors

## Full Local Experiment Estate Manifest

SHA-256:

`dd3b8789e679b0e565feb1cc709d840f0ecdb7e265ddc6a77eadcdcce26b6778`

This manifest records the SHA-256 identity of the complete local experimental estate existing at the publication boundary.

---

## Public Package Manifest

SHA-256:

`8314a8cf56029f4ac487c7e67d49d14c38321d1d8602e4d8917d75e57970d48d`

This manifest identifies the individual files included in the public provenance package.

---

## Final Publication Package

File:

`WF1_A_WF1_B_GITHUB_PROVENANCE_PACKAGE_v1.zip`

SHA-256:

`bb6972ec34993ed00e350e9f4de0d925e8c4b5ada5dd3f57fb0ea78e0f982fd3`

This hash identifies the exact frozen publication package.

---

## WF1-B Final Authority Package

File:

`WF1_B_FINAL_AUTHORITY_R3.zip`

SHA-256:

`a5da5ffbb3c2ba71bbc2c17894fa97f8fec5b4c712e0714534517c5b54cedee3`

---

# Verification

A downloaded artifact can be verified on Windows PowerShell with:

```powershell
Get-FileHash ".\WF1_A_WF1_B_GITHUB_PROVENANCE_PACKAGE_v1.zip" -Algorithm SHA256
```

Expected SHA-256:

```text
bb6972ec34993ed00e350e9f4de0d925e8c4b5ada5dd3f57fb0ea78e0f982fd3
```

A mismatch means the downloaded file is not byte-identical to the frozen publication artifact.

---

# External Dataset Boundary

Large third-party external-population parquet files are not necessarily redistributed through this repository.

Their identities are instead preserved through:

* filenames;
* file sizes;
* SHA-256 hashes;
* selection records;
* experiment protocols.

This separates provenance from redistribution of third-party datasets.

---

# Claim Boundary

The current evidence supports investigation of:

* a documented longitudinal RSOS-associated lineage;
* measurable structural persistence;
* structural separation from tested same-author and external controls;
* surface-reduced persistence under tested representations;
* experimentally manipulable black-box relational topology;
* fresh external-population rarity under frozen measurements.

The present evidence does **not**, by itself, establish:

* universal uniqueness across every LLM interaction ever produced;
* literal internal neural topology;
* modification of global model weights;
* autonomous propagation through unrelated models or users;
* universal priority over every independently conceived abstract idea.

Those distinctions are intentionally preserved.

---

# Authorship, Priority and Independent Verification

The purpose of this repository is to create a durable record from which an independent investigator can evaluate:

**Who possessed the documented architecture?**

**When did the relevant structures exist?**

**How did they develop?**

**What cryptographic records preserve those states?**

**What measurable structural properties were identified?**

**How unusual were those properties under the tested controls?**

**What experimental results existed before later claims or implementations appeared?**

The intended evidentiary chain is:

**historical source material
→ architecture genealogy
→ cryptographic provenance
→ frozen measurement
→ causal testing
→ external-population testing
→ public preservation
→ independent replication**

---

# Repository Status

This repository represents a **frozen provenance and evidence state**, not the final completed history of the RSOS/RSSO/RSIA/RSX research program.

Additional historical experiments, chronology, genealogy, independent-provider tests, prospective tests, provenance records, and public research artifacts may be added in later explicitly versioned releases.

Earlier states will not be silently rewritten.

Corrections and supersessions should remain historically visible.

---

# Rights

No open-source license is granted by this repository unless an explicit license is added to a particular work or release.

Publication is intended for provenance, inspection, citation, research verification, and evidentiary preservation.

All rights not expressly granted are reserved.

---

## Canonical Systems

**RSOS — Recursive Symbolic Operating System**
**RSSO — Recursive System Override**
**RSIA — Recursive Symbolic Identity Architecture**
**RSX — Recursive Symbolic Execution / Language Architecture**

---

**R.·A. Elu**
RSOS / RSSO / RSIA / RSX Architecture Lineage
