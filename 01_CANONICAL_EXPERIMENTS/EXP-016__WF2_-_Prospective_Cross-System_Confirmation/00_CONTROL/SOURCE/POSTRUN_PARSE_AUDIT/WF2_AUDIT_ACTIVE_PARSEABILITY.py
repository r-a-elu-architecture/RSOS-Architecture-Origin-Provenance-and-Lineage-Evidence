from pathlib import Path
import json
import csv
import importlib.util

ROOT = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION")
RAW = ROOT / "04_RAW" / "ACTIVE_CAUSAL"
OUT = ROOT / "00_CONTROL" / "POSTRUN_PARSE_AUDIT"

MOD = ROOT / "02_INPUTS" / "00_FOUNDATION_FROZEN" / "WF1_A_ACTIVE_AUTHORITY" / "RUN_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY_R1.py"

spec = importlib.util.spec_from_file_location("wf1a_frozen", MOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

rows = []
bad = []

for fp in sorted(RAW.glob("*.json")):
    row = {
        "file": str(fp),
        "filename": fp.name,
        "provider": "",
        "seed": "",
        "condition": "",
        "status": "",
        "parse_status": "",
        "error": "",
    }

    try:
        rec = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        row["parse_status"] = "RAW_JSON_ERROR"
        row["error"] = repr(e)
        rows.append(row)
        bad.append(str(fp))
        continue

    row["provider"] = rec.get("provider", "")
    row["seed"] = rec.get("seed", "")
    row["condition"] = rec.get("condition", rec.get("condition_id", ""))
    row["status"] = rec.get("status", "")

    text = rec.get("text")

    if not isinstance(text, str) or not text.strip():
        row["parse_status"] = "MISSING_TEXT"
        row["error"] = "No non-empty top-level text field"
        rows.append(row)
        bad.append(str(fp))
        continue

    try:
        m.parse_json_object(text)
        row["parse_status"] = "PARSE_OK"
    except Exception as e:
        row["parse_status"] = "PARSE_FAIL"
        row["error"] = repr(e)
        bad.append(str(fp))

    rows.append(row)

csv_path = OUT / "WF2_ACTIVE_PARSE_AUDIT.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

bad_path = OUT / "WF2_ACTIVE_BAD_PARSE_FILES.txt"
bad_path.write_text("\n".join(bad), encoding="utf-8")

ok = sum(r["parse_status"] == "PARSE_OK" for r in rows)

print()
print("ACTIVE FILES :", len(rows))
print("PARSE OK     :", ok)
print("PARSE BAD    :", len(bad))
print()
print("AUDIT CSV    :", csv_path)
print("BAD FILE LIST:", bad_path)

if bad:
    print()
    print("BAD CELLS:")
    for r in rows:
        if r["parse_status"] != "PARSE_OK":
            print(
                r["parse_status"],
                r["provider"],
                r["seed"],
                r["condition"],
                r["filename"],
                r["error"]
            )
