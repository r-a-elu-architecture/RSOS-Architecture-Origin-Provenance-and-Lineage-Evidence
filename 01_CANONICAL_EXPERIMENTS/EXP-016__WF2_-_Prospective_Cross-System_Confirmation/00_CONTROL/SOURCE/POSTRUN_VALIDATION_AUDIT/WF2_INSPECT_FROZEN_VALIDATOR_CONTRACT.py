from pathlib import Path
import importlib.util
import inspect

ROOT = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION")

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

print("=" * 80)
print("FROZEN validate_output()")
print("=" * 80)
print(inspect.getsource(m.validate_output))

print()
print("=" * 80)
print("FROZEN build_prompt()")
print("=" * 80)
print(inspect.getsource(m.build_prompt))
