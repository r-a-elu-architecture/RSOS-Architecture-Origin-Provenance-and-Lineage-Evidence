from pathlib import Path
import warnings
import joblib
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION")
MODELS = ROOT / "02_INPUTS" / "00_FOUNDATION_FROZEN" / "DETECTOR_PACKAGE" / "frozen_models"

warnings.simplefilter("error", InconsistentVersionWarning)

files = sorted(MODELS.glob("*.joblib"))

print("SCIKIT-LEARN:", sklearn.__version__)
print("MODEL FILES:", len(files))

for p in files:
    obj = joblib.load(p)
    print("LOAD_OK", p.name, type(obj).__name__)

print("ALL FROZEN MODELS LOAD: PASS")
