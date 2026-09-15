import csv
import json
import math
import re
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv_rows(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, fields=None):
    rows = list(rows)

    if fields is None:
        fields = list(rows[0].keys()) if rows else ["EMPTY"]

    with Path(path).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()

        if rows:
            w.writerows(rows)


def sha256_file(path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def verify_reference_rows(rows):
    checked = []

    for row in rows:
        p = Path(row["CANONICAL_FILE"])

        if not p.exists():
            raise RuntimeError(
                f"Canonical reference missing: {p}"
            )

        current = sha256_file(p)

        expected = row["SHA256"].lower()

        if current != expected:
            raise RuntimeError(
                f"Canonical reference SHA256 mismatch: {p}"
            )

        checked.append(row)

    return checked


def signed_log(x):
    if not np.isfinite(x):
        return np.nan

    return math.copysign(
        math.log1p(abs(float(x))),
        float(x),
    )


def numeric_stats(values):
    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if len(arr) == 0:
        return None

    qs = np.quantile(
        arr,
        [0.10, 0.25, 0.50, 0.75, 0.90],
    )

    return [
        math.log1p(len(arr)),
        float(np.mean(arr == 0.0)),
        signed_log(np.mean(arr)),
        math.log1p(float(np.std(arr))),
        signed_log(qs[0]),
        signed_log(qs[1]),
        signed_log(qs[2]),
        signed_log(qs[3]),
        signed_log(qs[4]),
        signed_log(float(np.min(arr))),
        signed_log(float(np.max(arr))),
    ]


def recursive_numeric_values(obj, output):
    if isinstance(obj, bool):
        return

    if isinstance(obj, (int, float)):
        if np.isfinite(float(obj)):
            output.append(float(obj))
        return

    if isinstance(obj, dict):
        for value in obj.values():
            recursive_numeric_values(
                value,
                output,
            )
        return

    if isinstance(obj, list):
        for value in obj:
            recursive_numeric_values(
                value,
                output,
            )


def numeric_variable_profiles(path):
    p = Path(path)
    ext = p.suffix.lower()

    profiles = []

    if ext == ".csv":
        df = pd.read_csv(
            p,
            low_memory=False,
        )

        for col in df.columns:
            numeric = pd.to_numeric(
                df[col],
                errors="coerce",
            ).to_numpy(dtype=float)

            stat = numeric_stats(
                numeric
            )

            if stat is not None:
                profiles.append(stat)

    elif ext == ".parquet":
        df = pd.read_parquet(p)

        for col in df.columns:
            try:
                numeric = pd.to_numeric(
                    df[col],
                    errors="coerce",
                ).to_numpy(dtype=float)
            except Exception:
                continue

            stat = numeric_stats(
                numeric
            )

            if stat is not None:
                profiles.append(stat)

    elif ext == ".json":
        obj = json.loads(
            p.read_text(
                encoding="utf-8-sig",
                errors="replace",
            )
        )

        values = []

        recursive_numeric_values(
            obj,
            values,
        )

        stat = numeric_stats(
            values
        )

        if stat is not None:
            profiles.append(stat)

    return profiles


def aggregate_profiles(profiles):
    if not profiles:
        return None

    arr = np.asarray(
        profiles,
        dtype=float,
    )

    q = np.quantile(
        arr,
        [0.10, 0.25, 0.50, 0.75, 0.90],
        axis=0,
    )

    vector = q.reshape(-1)

    extra = np.asarray(
        [
            math.log1p(arr.shape[0]),
            float(np.mean(np.isfinite(arr))),
        ],
        dtype=float,
    )

    return np.concatenate(
        [vector, extra]
    )


def robust_standardize(matrix):
    x = np.asarray(
        matrix,
        dtype=float,
    )

    median = np.nanmedian(
        x,
        axis=0,
    )

    mad = np.nanmedian(
        np.abs(x - median),
        axis=0,
    )

    scale = 1.4826 * mad

    fallback = np.nanstd(
        x,
        axis=0,
    )

    scale = np.where(
        scale > 1e-12,
        scale,
        fallback,
    )

    scale = np.where(
        scale > 1e-12,
        scale,
        1.0,
    )

    z = (
        x - median
    ) / scale

    z = np.where(
        np.isfinite(z),
        z,
        0.0,
    )

    return z


def euclidean_distance_matrix(matrix):
    x = np.asarray(
        matrix,
        dtype=float,
    )

    diff = (
        x[:, None, :]
        - x[None, :, :]
    )

    return np.sqrt(
        np.sum(
            diff * diff,
            axis=2,
        )
    )


def rankdata_average(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(values),
        dtype=float,
    )

    i = 0

    while i < len(values):
        j = i + 1

        while (
            j < len(values)
            and values[
                order[j]
            ]
            == values[
                order[i]
            ]
        ):
            j += 1

        rank = (
            (i + 1)
            + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = rank

        i = j

    return ranks


def spearman(x, y):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(x) < 2:
        return np.nan

    rx = rankdata_average(x)
    ry = rankdata_average(y)

    if (
        np.std(rx) <= 1e-12
        or np.std(ry) <= 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(
            rx,
            ry,
        )[0, 1]
    )


def recursive_named_numeric(obj, prefix="", output=None):
    if output is None:
        output = []

    if isinstance(obj, bool):
        return output

    if isinstance(obj, (int, float)):
        try:
            value = float(obj)
            if np.isfinite(value):
                output.append(
                    (prefix, value)
                )
        except Exception:
            pass

        return output

    if isinstance(obj, dict):
        for key, value in obj.items():
            child = (
                str(key)
                if not prefix
                else prefix + "." + str(key)
            )

            recursive_named_numeric(
                value,
                child,
                output,
            )

        return output

    if isinstance(obj, list):
        for value in obj:
            recursive_named_numeric(
                value,
                prefix,
                output,
            )

    return output


def named_numeric_from_file(path):
    p = Path(path)
    ext = p.suffix.lower()

    output = []

    if ext == ".csv":
        df = pd.read_csv(
            p,
            low_memory=False,
        )

        for col in df.columns:
            numeric = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            for value in numeric.dropna():
                value = float(value)

                if np.isfinite(value):
                    output.append(
                        (str(col), value)
                    )

    elif ext == ".json":
        obj = json.loads(
            p.read_text(
                encoding="utf-8-sig",
                errors="replace",
            )
        )

        output.extend(
            recursive_named_numeric(
                obj
            )
        )

    return output
