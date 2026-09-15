from pathlib import Path
import sys
import warnings
import joblib
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION")
MODELS = (
    ROOT
    / "02_INPUTS"
    / "00_FOUNDATION_FROZEN"
    / "DETECTOR_PACKAGE"
    / "frozen_models"
)

warnings.simplefilter("error", InconsistentVersionWarning)

print("RUNTIME_SKLEARN =", sklearn.__version__)
print()

for p in sorted(MODELS.glob("*.joblib")):
    try:
        obj = joblib.load(p)
        print(f"LOAD_OK\t{p.name}\t{type(obj).__name__}")
    except InconsistentVersionWarning as e:
        print(f"VERSION_MISMATCH\t{p.name}\t{e}")
    except Exception as e:
        print(f"OTHER_ERROR\t{p.name}\t{type(e).__name__}: {e}")

