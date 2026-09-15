import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

model_path = Path(sys.argv[1])

model = joblib.load(model_path)

feature_names = list(
    map(
        str,
        getattr(model, "feature_names_in_", [])
    )
)

print(
    json.dumps({
        "ready": True,
        "model": model_path.name,
        "feature_names_in": feature_names,
        "n_features_in": int(
            getattr(model, "n_features_in_", len(feature_names))
        )
    }),
    flush=True
)

for line in sys.stdin:

    line = line.strip()

    if not line:
        continue

    try:
        req = json.loads(line)

        if req.get("op") == "decision_function":

            df = pd.DataFrame(req["rows"])

            cols = req.get("columns")

            if cols:
                df = df[cols]

            values = np.asarray(
                model.decision_function(df),
                dtype=float
            ).reshape(-1)

            print(
                json.dumps({
                    "ok": True,
                    "values": values.tolist()
                }),
                flush=True
            )

        elif req.get("op") == "close":

            print(
                json.dumps({"ok": True}),
                flush=True
            )
            break

        else:
            print(
                json.dumps({
                    "ok": False,
                    "error": "UNKNOWN_OPERATION"
                }),
                flush=True
            )

    except Exception as e:

        print(
            json.dumps({
                "ok": False,
                "error": repr(e)
            }),
            flush=True
        )
