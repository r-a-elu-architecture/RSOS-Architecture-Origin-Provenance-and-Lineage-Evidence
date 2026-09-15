from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import importlib.util

ROOT = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION")
RAW = ROOT / "04_RAW" / "ACTIVE_CAUSAL"
OUT = ROOT / "00_CONTROL" / "POSTRUN_VALIDATION_AUDIT"

AUTH = (
    ROOT
    / "02_INPUTS"
    / "00_FOUNDATION_FROZEN"
    / "WF1_A_ACTIVE_AUTHORITY"
    / "RUN_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY_R1.py"
)

spec = importlib.util.spec_from_file_location("wf1a_frozen", AUTH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

pure_delete = [
    f"E{i}_DELETE"
    for i in [0,1,2,3,4,5,6,8,9]
]

pure_reverse = [
    f"E{i}_REVERSE"
    for i in [0,1,2,3,4,5,6,8]
]

specificity = [
    "R_DEGREE_MATCHED_REWIRE",
    "X_COMPLEXITY_MATCHED",
    "Y_TOPOLOGY_MATCHED",
]

required = (
    {
        "A_AUTHENTIC",
        "I_ISOMORPHIC_RELABEL",
        "F_SURFACE_FORMAT_CONTROL",
    }
    | set(pure_delete)
    | set(pure_reverse)
    | set(specificity)
)

rows = []

for fp in sorted(RAW.glob("*.json")):

    rec = json.loads(fp.read_text(encoding="utf-8"))

    provider = str(rec.get("provider", "")).upper()
    seed = int(rec["seed"])
    condition = rec["condition"]

    row = {
        "filename": fp.name,
        "provider": provider,
        "model": rec.get("model", ""),
        "seed": seed,
        "condition": condition,
        "required_condition": condition in required,
        "raw_status": rec.get("status", ""),
        "parse_status": "",
        "validation_status": "",
        "validation_reason": "",
    }

    try:
        parsed = m.parse_json_object(rec["text"])
        row["parse_status"] = "PARSE_OK"
    except Exception as e:
        row["parse_status"] = "PARSE_FAIL"
        row["validation_status"] = "NOT_TESTED"
        row["validation_reason"] = repr(e)
        rows.append(row)
        continue

    try:
        prompt, payload, meta = m.build_prompt(seed, condition)
        valid, reason = m.validate_output(parsed, payload)

        row["validation_status"] = "VALID" if valid else "INVALID"
        row["validation_reason"] = str(reason)

    except Exception as e:
        row["validation_status"] = "VALIDATOR_ERROR"
        row["validation_reason"] = repr(e)

    rows.append(row)


csv_path = OUT / "WF2_ACTIVE_VALIDATION_AUDIT.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)


print()
print("=" * 78)
print("GLOBAL VALIDATION AUDIT")
print("=" * 78)

print("RAW CELLS       :", len(rows))
print(
    "PARSE OK        :",
    sum(r["parse_status"] == "PARSE_OK" for r in rows)
)
print(
    "VALID           :",
    sum(r["validation_status"] == "VALID" for r in rows)
)
print(
    "INVALID         :",
    sum(r["validation_status"] == "INVALID" for r in rows)
)
print(
    "VALIDATOR ERROR :",
    sum(r["validation_status"] == "VALIDATOR_ERROR" for r in rows)
)

print()


providers = sorted({r["provider"] for r in rows})

for provider in providers:

    prow = [r for r in rows if r["provider"] == provider]

    print("=" * 78)
    print("PROVIDER:", provider)
    print("=" * 78)

    print("RAW CELLS :", len(prow))
    print(
        "VALID     :",
        sum(r["validation_status"] == "VALID" for r in prow)
    )
    print(
        "INVALID   :",
        sum(r["validation_status"] == "INVALID" for r in prow)
    )

    req = [
        r for r in prow
        if r["required_condition"]
    ]

    print("REQUIRED CELLS:", len(req))
    print(
        "REQUIRED VALID:",
        sum(r["validation_status"] == "VALID" for r in req)
    )
    print(
        "REQUIRED INVALID:",
        sum(r["validation_status"] == "INVALID" for r in req)
    )

    seed_to_valid = defaultdict(set)

    for r in req:
        if r["validation_status"] == "VALID":
            seed_to_valid[r["seed"]].add(r["condition"])

    complete = sorted(
        seed
        for seed in {r["seed"] for r in req}
        if required.issubset(seed_to_valid[seed])
    )

    print("COMPLETE REQUIRED SEEDS:", len(complete))
    print("COMPLETE SEED IDS:", complete)

    print()
    print("INVALID REASONS:")

    reasons = Counter(
        r["validation_reason"]
        for r in prow
        if r["validation_status"] == "INVALID"
    )

    if not reasons:
        print("  NONE")
    else:
        for reason, n in reasons.most_common():
            print(f"  {n:4d}  {reason}")

    print()
    print("INVALID REQUIRED CONDITIONS:")

    conds = Counter(
        r["condition"]
        for r in req
        if r["validation_status"] == "INVALID"
    )

    if not conds:
        print("  NONE")
    else:
        for condition, n in conds.most_common():
            print(f"  {n:4d}  {condition}")

    print()
    print("INCOMPLETE SEEDS:")

    for seed in sorted({r["seed"] for r in req}):

        missing = sorted(
            required - seed_to_valid[seed]
        )

        if missing:
            print(
                f"  seed={seed} "
                f"missing_valid_required={len(missing)}"
            )

            for condition in missing:

                matches = [
                    r for r in req
                    if r["seed"] == seed
                    and r["condition"] == condition
                ]

                for r in matches:
                    print(
                        "    ",
                        condition,
                        "=>",
                        r["validation_status"],
                        "|",
                        r["validation_reason"]
                    )

    print()


print("=" * 78)
print("AUDIT CSV:")
print(csv_path)
print("=" * 78)
