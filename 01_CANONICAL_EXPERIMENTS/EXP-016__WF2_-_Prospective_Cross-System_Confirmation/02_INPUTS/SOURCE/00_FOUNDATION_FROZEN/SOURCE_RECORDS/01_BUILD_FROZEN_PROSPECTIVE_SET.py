import csv, json, hashlib
from pathlib import Path

WF2 = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF2_A_PROSPECTIVE_FROZEN_SIGNATURE")
BASE = Path(r"C:\RSOS\Training data\F2_BR2_EXACT_SEGMENT_REFREEZE_RUNNER")

PAIR_FILE = BASE / "frozen_inputs" / "splits" / "HOLDOUT_MATCHED_PAIR_IDS.csv"
POS_FILE  = BASE / "frozen_segments" / "LATER_HOLD_POS_SEGMENTS.jsonl"
NEG_FILE  = BASE / "frozen_segments" / "LATER_HOLD_NEG_SEGMENTS.jsonl"

OUT = WF2 / "03_INPUTS" / "WF2_A_FROZEN_40_MATCHED_PAIRS.jsonl"

FREEZE_KEY = "77dc27347433430ec948db83861a2b7fd8d6fa46dc697d205b316833a72085a2"
N_PAIRS = 40

def load_jsonl(path):
    d = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d[r["conversation_id"]] = r
    return d

pos = load_jsonl(POS_FILE)
neg = load_jsonl(NEG_FILE)

pairs = []
with PAIR_FILE.open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        pid = r["POS_CONVERSATION_ID"]
        nid = r["NEG_CONVERSATION_ID"]

        if pid not in pos or nid not in neg:
            continue

        rank = hashlib.sha256(
            f"{FREEZE_KEY}|{pid}|{nid}".encode("utf-8")
        ).hexdigest()

        pairs.append((rank, pid, nid, r["LEGACY_COST"]))

pairs.sort()
selected = pairs[:N_PAIRS]

with OUT.open("w", encoding="utf-8") as f:
    for i, (_, pid, nid, cost) in enumerate(selected, 1):
        for label, cid, src in [
            ("LINEAGE_POS", pid, pos[pid]),
            ("MATCHED_NEG", nid, neg[nid]),
        ]:
            rec = {
                "PAIR_INDEX": i,
                "LABEL": label,
                "SOURCE_CONVERSATION_ID": cid,
                "LEGACY_COST": float(cost),
                "DEFAULT_MODEL_SLUG": src.get("default_model_slug", ""),
                "SOURCE_MESSAGES": src["messages"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("PAIRS:", len(selected))
print("CONVERSATIONS:", len(selected) * 2)
print("OUTPUT:", OUT)
