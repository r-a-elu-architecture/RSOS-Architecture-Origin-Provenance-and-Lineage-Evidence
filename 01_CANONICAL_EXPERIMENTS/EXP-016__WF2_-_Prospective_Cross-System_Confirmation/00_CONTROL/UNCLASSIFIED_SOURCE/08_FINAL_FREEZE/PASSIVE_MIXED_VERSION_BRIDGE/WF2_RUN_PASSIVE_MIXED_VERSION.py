from pathlib import Path
import atexit
import importlib.util
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION"
)

ORIGINAL = (
    ROOT
    / "03_RUNNER"
    / "WF2_SCORE_AND_ADJUDICATE.py"
)

WORKER = (
    ROOT
    / "00_CONTROL"
    / "WF2_MIXED_VERSION_BRIDGE"
    / "WF2_SKLEARN18_MODEL_WORKER.py"
)

PY18 = (
    ROOT
    / ".venv_wf2_score"
    / "Scripts"
    / "python.exe"
)

REMOTE_18 = {
    "R1_COMPATIBLE.joblib",
    "SURFACE_FREE.joblib",
}

_original_joblib_load = joblib.load
_workers = []


class RemoteModel:

    def __init__(self, path):

        self.path = Path(path)

        self.proc = subprocess.Popen(
            [
                str(PY18),
                "-u",
                str(WORKER),
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        line = self.proc.stdout.readline()

        if not line:
            err = self.proc.stderr.read()
            raise RuntimeError(
                f"Remote model failed to start: "
                f"{self.path.name}\n{err}"
            )

        meta = json.loads(line)

        if not meta.get("ready"):
            raise RuntimeError(
                f"Remote model initialization failed: {meta}"
            )

        self.feature_names_in_ = np.asarray(
            meta.get("feature_names_in", []),
            dtype=object
        )

        self.n_features_in_ = int(
            meta.get(
                "n_features_in",
                len(self.feature_names_in_)
            )
        )

        _workers.append(self)

        print(
            "NATIVE_MODEL_ROUTE",
            self.path.name,
            "=> sklearn 1.8.0",
            flush=True
        )

    def decision_function(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        rows = json.loads(
            X.to_json(
                orient="records",
                double_precision=15
            )
        )

        req = {
            "op": "decision_function",
            "columns": list(map(str, X.columns)),
            "rows": rows,
        }

        self.proc.stdin.write(
            json.dumps(req) + "\n"
        )
        self.proc.stdin.flush()

        line = self.proc.stdout.readline()

        if not line:
            err = self.proc.stderr.read()
            raise RuntimeError(
                f"Remote scorer died for {self.path.name}\n{err}"
            )

        result = json.loads(line)

        if not result.get("ok"):
            raise RuntimeError(
                f"Remote scoring error for "
                f"{self.path.name}: {result}"
            )

        return np.asarray(
            result["values"],
            dtype=float
        )

    def close(self):

        if self.proc.poll() is not None:
            return

        try:
            self.proc.stdin.write(
                json.dumps({"op": "close"}) + "\n"
            )
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)

        except Exception:
            self.proc.kill()


def shutdown():

    for w in list(_workers):
        try:
            w.close()
        except Exception:
            pass


atexit.register(shutdown)


def routed_joblib_load(path, *args, **kwargs):

    name = Path(path).name

    if name in REMOTE_18:
        return RemoteModel(path)

    obj = _original_joblib_load(
        path,
        *args,
        **kwargs
    )

    if name.endswith(".joblib"):
        print(
            "NATIVE_MODEL_ROUTE",
            name,
            "=> sklearn 1.9.1",
            flush=True
        )

    return obj


# Route only frozen model deserialization.
joblib.load = routed_joblib_load


spec = importlib.util.spec_from_file_location(
    "wf2_original_adjudicator",
    ORIGINAL
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.main()
