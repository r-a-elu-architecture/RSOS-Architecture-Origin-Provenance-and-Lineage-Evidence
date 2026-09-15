from pathlib import Path
import json
import importlib.util

ROOT = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION")

AUTH = (
    ROOT
    / "02_INPUTS"
    / "00_FOUNDATION_FROZEN"
    / "WF1_A_ACTIVE_AUTHORITY"
    / "RUN_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY_R1.py"
)

RAW = ROOT / "04_RAW" / "ACTIVE_CAUSAL"

spec = importlib.util.spec_from_file_location("wf1a_frozen", AUTH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print("=" * 90)
print("FROZEN INSTRUCTION")
print("=" * 90)
print(m.INSTRUCTION)

print()
print("=" * 90)
print("INVALID PLUS EXAMPLES")
print("=" * 90)

shown = {
    "ANTHROPIC": 0,
    "OPENAI": 0,
    "GOOGLE": 0,
}

for fp in sorted(RAW.glob("*.json")):

    rec = json.loads(fp.read_text(encoding="utf-8"))

    provider = str(rec.get("provider", "")).upper()

    if provider not in shown or shown[provider] >= 5:
        continue

    seed = int(rec["seed"])
    condition = rec["condition"]

    parsed = m.parse_json_object(rec["text"])

    prompt, payload, meta = m.build_prompt(seed, condition)
    valid, reason = m.validate_output(parsed, payload)

    if valid or reason != "PLUS":
        continue

    V = set(payload["V"])
    active_edges = m.active_label_edges(payload)

    try:
        plus_parsed = m.parse_node_pairs(parsed["plus"], V)
    except Exception as e:
        plus_parsed = f"PARSE_ERROR: {e!r}"

    print()
    print("-" * 90)
    print("PROVIDER :", provider)
    print("SEED     :", seed)
    print("CONDITION:", condition)
    print("FILE     :", fp.name)
    print("RAW PLUS :", parsed.get("plus"))
    print("PARSED   :", plus_parsed)
    print("ACTIVE EDGES:")
    print(sorted(active_edges))
    print("VALIDATOR:", reason)

    shown[provider] += 1

    if all(v >= 5 or k == "GOOGLE" and v >= 3 for k,v in shown.items()):
        pass

print()
print("=" * 90)
print("COUNTS SHOWN:", shown)
print("=" * 90)
