from pathlib import Path
import argparse, csv, hashlib, json, math, os, platform, re, shutil, sys, time, zipfile, zlib
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

VERSION = "F2-A_WILDCHAT_3SHARD_EXTERNAL_POPULATION_RARITY"
SEED = 120918
RNG = np.random.default_rng(SEED)

OFFICIAL_SHARDS = {
    "train-00049-of-00086.parquet": (103170595, "6f8f300c0501ce5d83912abf1ae1097ed04c30d9ef93005003a39f4c61f1024c"),
    "train-00080-of-00086.parquet": (186001659, "1ed63ac322419a05307a474c0db595cb440ca8d72c434dfcfd74f4bdf90028ae"),
    "train-00085-of-00086.parquet": (496266956, "898fd913b72075c13e237a70e9f60da9adbb9fd652b46970e123a3258b19f9ff"),
}
F1_REQUIRED_SHA256 = "ce0191cdbc627c773c360d92e35975302d721b18f188fdb4689e80ad529108b4"
BR1_REQUIRED_SHA256 = "159b6a0df406073be5bdf9b242cdd45fde0d7b6281252cef813b88cc8971c364"
MIN_CHARS = 800
MIN_TURNS = 3
BOOT = 500

DIRECT_RE = re.compile(r'''(?ix)\b(?:RSOS|RSSO|RSIA|RSX|CM1|LINX|MEMSTACK|CONTAIN|UGCL|CML|SRC[-\s]?X(?:\.5)?|R\.\s*A\.\s*Elu)\b|recursive\s+symbolic\s+(?:operating\s+system|system\s+operating|identity\s+architecture|(?:eXpression|expression)(?:\s+language)?)|\b(?:tier(?:s)?|glyph(?:s)?|archetype(?:s)?|override(?:s|d|ing)?|containment|resurrection|echo\s+protocol|activation\s+protocol)\b|\b(?:lion|dragon|eagle|whale|serpent)\b''')

CHAR_EDGES = np.array([800,1600,3200,6400,12800,25600,51200,102400,204800,409600,819200,1e12], float)
TURN_EDGES = np.array([3,5,8,13,21,34,55,89,144,233,377,1e6], float)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def redact(text: str):
    return DIRECT_RE.sub(lambda m: " " * len(m.group(0)), text)


def safe(a, b):
    return float(a / b) if b else 0.0


def entropy(vals):
    if not vals:
        return 0.0
    bins = [0, 80, 300, 1000, 3000, 10000, 10**9]
    c = np.histogram(vals, bins=bins)[0]
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def deterministic_features(messages):
    turns = []
    redacted_spans = 0
    rawchars = 0
    for m in messages:
        t = str(m.get("text", "") or "")
        rawchars += len(t)
        bt = redact(t)
        redacted_spans += sum(1 for _ in DIRECT_RE.finditer(t))
        if bt.strip():
            turns.append({"role": str(m.get("role", "") or ""), "text": bt})
    texts = [t["text"] for t in turns]
    roles = [t["role"] for t in turns]
    full = "\n".join(texts)
    low = full.lower()
    total = len(full)
    toks = re.findall(r"[a-z0-9_]+", low)
    ctr = Counter(toks)
    lens = [len(x) for x in texts]
    n = len(turns)
    user = "\n".join(t["text"] for t in turns if t["role"] == "user")
    ass = "\n".join(t["text"] for t in turns if t["role"] == "assistant")
    ns = sum(1 for c in full if not c.isspace())
    up = sum(1 for c in full if c.isupper())
    dg = sum(1 for c in full if c.isdigit())
    roletrans = sum(1 for a, b in zip(roles, roles[1:]) if a != b)
    dup = (len(texts) - len(set(texts))) if texts else 0
    para = sum(max(1, x.count("\n") + 1) for x in texts)
    fences = full.count("```") // 2
    heads = re.findall(r"(?m)^\s*(#{1,6})\s", full)
    listn = len(re.findall(r"(?m)^\s*(?:[-*+] |\d+[.)] )", full))
    br = sum(full.count(x) for x in "[]{}()")
    enc = full.encode("utf-8", errors="ignore")
    comp = len(zlib.compress(enc, 6)) / len(enc) if enc else 0.0
    mean = float(np.mean(lens)) if lens else 0.0
    mx = max(lens, default=0)
    f = {
        "REDACTED_SPANS": redacted_spans,
        "N_TURNS": n,
        "TOTAL_CHARS": total,
        "TYPE_TOKEN_RATIO": safe(len(ctr), len(toks)),
        "UPPERCASE_RATIO": safe(up, ns),
        "DIGIT_RATIO": safe(dg, ns),
        "COMPRESSION_RATIO": comp,
        "USER_CHAR_SHARE": safe(len(user), total),
        "MEAN_MAX_TURN_RATIO": safe(mean, mx),
        "WORDS_PER_1K_CHAR": 1000 * safe(len(toks), total),
        "TURNS_PER_1K_CHAR": 1000 * safe(n, total),
        "PARAGRAPHS_PER_TURN": safe(para, n),
        "CODE_FENCES_PER_1K_CHAR": 1000 * safe(fences, total),
        "HEADERS_PER_1K_CHAR": 1000 * safe(len(heads), total),
        "LIST_ITEMS_PER_1K_CHAR": 1000 * safe(listn, total),
        "COLONS_PER_1K_CHAR": 1000 * safe(full.count(":"), total),
        "BRACKETS_PER_1K_CHAR": 1000 * safe(br, total),
        "ROLE_ALTERNATION_RATIO": safe(roletrans, max(1, n - 1)) if n else 0.0,
        "DUPLICATE_TURN_RATIO": safe(dup, n),
        "USER_ASSISTANT_CHAR_RATIO": safe(len(user), len(ass)),
        "TURN_CHAR_CV": safe(float(np.std(lens)), mean) if lens else 0.0,
        "TURN_LENGTH_ENTROPY": entropy(lens),
        "QUESTION_PER_1K_CHAR": 1000 * safe(full.count("?"), total),
        "EXCLAM_PER_1K_CHAR": 1000 * safe(full.count("!"), total),
        "EQUAL_PER_1K_CHAR": 1000 * safe(full.count("="), total),
        "SEMICOLON_PER_1K_CHAR": 1000 * safe(full.count(";"), total),
        "PIPE_PER_1K_CHAR": 1000 * safe(full.count("|"), total),
        "HEADER_LEVEL_MEAN": float(np.mean([len(x) for x in heads])) if heads else 0.0,
        "HEADER_LEVEL_MAX": float(max([len(x) for x in heads], default=0)),
        "USER_TURN_MEAN": 0.0,
        "ASSISTANT_TURN_MEAN": 0.0,
        "RESPONSE_EXPANSION_RATIO": 0.0,
    }
    ul = [len(t["text"]) for t in turns if t["role"] == "user"]
    al = [len(t["text"]) for t in turns if t["role"] == "assistant"]
    f["USER_TURN_MEAN"] = float(np.mean(ul)) if ul else 0.0
    f["ASSISTANT_TURN_MEAN"] = float(np.mean(al)) if al else 0.0
    f["RESPONSE_EXPANSION_RATIO"] = safe(
        float(np.mean(al)) if al else 0.0,
        float(np.mean(ul)) if ul else 0.0,
    )
    return f, rawchars, len(messages)


def portable_score(df, portable, key):
    m = portable["models"][key]
    cols = m["features"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{key} missing extracted features: {missing}")
    X = df[cols].to_numpy(float)
    mean = np.asarray(m["mean"], float)
    scale = np.asarray(m["scale"], float)
    coef = np.asarray(m["coef"], float)
    intercept = float(m["intercept"])
    z = ((X - mean) / scale) @ coef + intercept
    return 1 / (1 + np.exp(-np.clip(z, -709, 709)))


def auc_fast(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    ns = np.sort(neg)
    left = np.searchsorted(ns, pos, side="left")
    right = np.searchsorted(ns, pos, side="right")
    return float(np.mean((left + 0.5 * (right - left)) / len(ns)))


def auc_ci(pos, neg, nboot=BOOT):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    auc = auc_fast(pos, neg)
    boots = np.empty(nboot, float)
    for i in range(nboot):
        pp = pos[RNG.integers(0, len(pos), len(pos))]
        nn = neg[RNG.integers(0, len(neg), len(neg))]
        boots[i] = auc_fast(pp, nn)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = float(stats.mannwhitneyu(pos, neg, alternative="greater").pvalue)
    return auc, float(lo), float(hi), p


def clopper_pearson(k, n, alpha=0.05):
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def weighted_mean(x, w):
    return float(np.sum(np.asarray(x, float) * np.asarray(w, float)) / np.sum(w))


def weighted_var(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = weighted_mean(x, w)
    return float(np.sum(w * (x - m) ** 2) / np.sum(w))


def weighted_smd(x1, w1, x0, w0):
    m1, m0 = weighted_mean(x1, w1), weighted_mean(x0, w0)
    v1, v0 = weighted_var(x1, w1), weighted_var(x0, w0)
    den = math.sqrt(max((v1 + v0) / 2.0, 1e-12))
    return float((m1 - m0) / den)


def normalize_messages(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        if item is None:
            continue
        if hasattr(item, "as_py"):
            item = item.as_py()
        if not isinstance(item, dict):
            try:
                item = dict(item)
            except Exception:
                continue
        role = str(item.get("role", "") or "").lower().strip()
        text = item.get("content", item.get("text", ""))
        if text is None:
            text = ""
        text = str(text)
        if role in {"human"}:
            role = "user"
        elif role in {"ai", "bot"}:
            role = "assistant"
        out.append({"role": role, "text": text})
    return out


def blind_id(conversation_hash, messages):
    base = str(conversation_hash or "")
    if not base:
        base = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(("F2-A|" + base).encode("utf-8", errors="ignore")).hexdigest()


def find_by_hash(search_roots, filename, expected_sha):
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob(filename):
            if p.is_file():
                candidates.append(p)
    # Prefer canonical training-data source, then shortest path.
    candidates = sorted(set(candidates), key=lambda p: ("training data\\training data" not in str(p).lower(), len(str(p))))
    for p in candidates:
        if sha256_file(p).lower() == expected_sha.lower():
            return p
    return None


def locate_br1(search_roots):
    return find_by_hash(search_roots, "F1_BR1_BLIND_RSOS_CORPUS_REDISCOVERY_RESULTS_PACKAGE.zip", BR1_REQUIRED_SHA256)


def extract_br1_tables(br1_zip, dst):
    dst.mkdir(parents=True, exist_ok=True)
    wanted = {
        "F1_BR1_RESULTS_PACKAGE/05_FEATURES/LATER_HOLD_POS_FEATURES.csv": "LATER_HOLD_POS_FEATURES.csv",
        "F1_BR1_RESULTS_PACKAGE/07_RESULTS/HOLDOUT_SCORES_INTERNAL.csv": "HOLDOUT_SCORES_INTERNAL.csv",
        "F1_BR1_RESULTS_PACKAGE/06_MODELS/R1_COMPATIBLE.joblib": "R1_COMPATIBLE.joblib",
        "F1_BR1_RESULTS_PACKAGE/06_MODELS/SURFACE_FREE.joblib": "SURFACE_FREE.joblib",
    }
    with zipfile.ZipFile(br1_zip) as z:
        names = set(z.namelist())
        for src, out in wanted.items():
            if src not in names:
                raise RuntimeError(f"Missing frozen BR1 member: {src}")
            (dst / out).write_bytes(z.read(src))


def assign_bins(df):
    d = df.copy()
    d["CHAR_BIN"] = pd.cut(pd.to_numeric(d["raw_chars"], errors="coerce"), CHAR_EDGES, right=False, labels=False, include_lowest=True)
    d["TURN_BIN"] = pd.cut(pd.to_numeric(d["raw_turns"], errors="coerce"), TURN_EDGES, right=False, labels=False, include_lowest=True)
    d["CODE_FLAG"] = (pd.to_numeric(d["CODE_FENCES_PER_1K_CHAR"], errors="coerce").fillna(0) > 0).astype(int)
    d["CEM_CELL"] = d["CHAR_BIN"].astype("Int64").astype(str) + "|" + d["TURN_BIN"].astype("Int64").astype(str) + "|" + d["CODE_FLAG"].astype(str)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\RSOS\Training data\F2_EXTERNAL_POPULATION_RARITY")
    ap.add_argument("--f1-zip", default=r"C:\RSOS\F1_FINAL_COMPLETE_LOCAL.zip")
    ap.add_argument("--search-root", action="append", default=[])
    args = ap.parse_args()

    root = Path(args.root)
    search_roots = [Path(x) for x in args.search_root] or [
        Path(r"C:\RSOS\Training data\training data"),
        Path(r"C:\Users\user\Documents\RSOS\chatgpt 20260611\Prompting Structure"),
        Path(r"C:\RSOS"),
        Path(r"C:\Users\user\Documents\RSOS"),
    ]

    dirs = {
        "inputs": root / "00_INPUTS_WILDCHAT",
        "frozen": root / "01_FROZEN_IMPLEMENTATION",
        "manifest": root / "02_INPUT_MANIFEST",
        "schema": root / "03_SCHEMA",
        "features": root / "04_EXTERNAL_FEATURES",
        "results": root / "05_PRIMARY_RESULTS",
        "rarity": root / "06_RARITY_TAILS",
        "nuisance": root / "07_NUISANCE_CONTROL",
        "nearest": root / "08_NEAREST_CONTROLS",
        "ledger": root / "09_STATUS_LEDGER",
        "final": root / "10_FINAL_MANIFEST",
        "references": root / "11_F1_REFERENCES",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    marker = root / "F2_A_COMPLETE.marker"
    if marker.exists():
        raise SystemExit(f"F2-A already completed; refusing overwrite: {marker}")

    script_dir = Path(__file__).resolve().parent
    portable_src = script_dir / "PORTABLE_FROZEN_BR1_MODELS.json"
    spec_src = script_dir / "F2_FROZEN_IMPLEMENTATION_SPEC.json"
    if not portable_src.exists() or not spec_src.exists():
        raise FileNotFoundError("Runner package missing portable model or frozen implementation spec.")
    shutil.copy2(portable_src, dirs["frozen"] / portable_src.name)
    shutil.copy2(spec_src, dirs["frozen"] / spec_src.name)
    shutil.copy2(Path(__file__), dirs["frozen"] / Path(__file__).name)

    # 0. Verify F1 immutable boundary BEFORE any reserved 00086 file is opened.
    f1zip = Path(args.f1_zip)
    if not f1zip.exists():
        raise FileNotFoundError(f"Missing F1 local freeze: {f1zip}")
    f1hash = sha256_file(f1zip).lower()
    if f1hash != F1_REQUIRED_SHA256:
        raise RuntimeError(f"F1 FINAL HASH MISMATCH: {f1hash} != {F1_REQUIRED_SHA256}. F2 aborted before reserved data read.")
    (dirs["manifest"] / "F1_FINAL_VERIFIED.txt").write_text(f"{f1hash}  {f1zip}\n", encoding="ascii")

    # Verify frozen BR1 reference package BEFORE external content analysis.
    br1zip = locate_br1(search_roots)
    if br1zip is None:
        raise RuntimeError("Could not find hash-verified BR1 results package. F2 aborted before content analysis.")
    br1hash = sha256_file(br1zip).lower()
    if br1hash != BR1_REQUIRED_SHA256:
        raise RuntimeError("BR1 reference package hash mismatch.")
    shutil.copy2(br1zip, dirs["references"] / br1zip.name)
    extract_br1_tables(br1zip, dirs["references"] / "EXTRACTED")

    portable = json.loads(portable_src.read_text(encoding="utf-8"))
    pos = pd.read_csv(dirs["references"] / "EXTRACTED" / "LATER_HOLD_POS_FEATURES.csv")
    stored = pd.read_csv(dirs["references"] / "EXTRACTED" / "HOLDOUT_SCORES_INTERNAL.csv")
    stored_pos = stored[pd.to_numeric(stored["y"], errors="coerce").eq(1)].copy()

    provenance = []
    for key in ["R1_COMPATIBLE", "SURFACE_FREE"]:
        job = dirs["references"] / "EXTRACTED" / f"{key}.joblib"
        jhash = sha256_file(job)
        exp = portable["models"][key]["source_joblib_sha256"]
        if jhash != exp:
            raise RuntimeError(f"Frozen {key} joblib hash mismatch: {jhash} != {exp}")
        pos[f"{key}_SCORE"] = portable_score(pos, portable, key)
        smap = dict(zip(stored_pos["conversation_id"].astype(str), stored_pos[f"{key}_SCORE"].astype(float)))
        expected = np.array([smap[str(x)] for x in pos["conversation_id"]], float)
        md = float(np.max(np.abs(pos[f"{key}_SCORE"].to_numpy(float) - expected)))
        if md >= 1e-10:
            raise RuntimeError(f"Portable frozen score equivalence failed for {key}: {md}")
        provenance.append({"MODEL": key, "JOBLIB_SHA256": jhash, "MAX_ABS_DIFF_VS_FROZEN_BR1": md, "PASS": True})
    pd.DataFrame(provenance).to_csv(dirs["manifest"] / "FROZEN_MODEL_EQUIVALENCE.csv", index=False)

    thresholds = []
    for key in ["SURFACE_FREE", "R1_COMPATIBLE"]:
        a = pos[f"{key}_SCORE"].to_numpy(float)
        for label, q in [("MEDIAN", .50), ("Q75", .75), ("Q90", .90), ("MAX", 1.0)]:
            thresholds.append({"MODEL": key, "REFERENCE": label, "QUANTILE": q, "THRESHOLD": float(np.quantile(a, q)), "N_LINEAGE": len(a)})
    pd.DataFrame(thresholds).to_csv(dirs["manifest"] / "FROZEN_LINEAGE_THRESHOLDS_PRE_EXTERNAL.csv", index=False)

    # 1. Locate/hash/copy the three exact official WildChat-4.8M shards.
    # Hashing is the first permitted reserved-data byte read after F1 freeze verification.
    (root / "F2_A_OPENED.marker").write_text(f"{utcnow()}\nF1={f1hash}\n", encoding="ascii")
    shard_rows = []
    staged = []
    for name, (exp_bytes, exp_sha) in OFFICIAL_SHARDS.items():
        found = find_by_hash(search_roots, name, exp_sha)
        if found is None:
            raise RuntimeError(f"Official WildChat shard not found with expected SHA256: {name}")
        if found.stat().st_size != exp_bytes:
            raise RuntimeError(f"Official WildChat shard size mismatch: {name}")
        dst = dirs["inputs"] / name
        if not dst.exists() or sha256_file(dst).lower() != exp_sha.lower():
            shutil.copy2(found, dst)
        got = sha256_file(dst).lower()
        if got != exp_sha.lower():
            raise RuntimeError(f"Staged WildChat hash mismatch: {name}")
        staged.append(dst)
        shard_rows.append({"FILE": name, "SOURCE_PATH": str(found), "STAGED_PATH": str(dst), "BYTES": dst.stat().st_size, "SHA256": got, "OFFICIAL_HASH_MATCH": True})
    pd.DataFrame(shard_rows).to_csv(dirs["manifest"] / "WILDCHAT_OFFICIAL_SHARD_MANIFEST.csv", index=False)

    # 2. Stream parquet rows; never emit raw conversation text.
    rows = []
    errors = []
    raw_seen = 0
    for shard in staged:
        pf = pq.ParquetFile(shard)
        schema_names = pf.schema_arrow.names
        required = {"conversation", "model"}
        if not required.issubset(schema_names):
            raise RuntimeError(f"Unexpected WildChat schema in {shard.name}: {schema_names}")
        meta_cols = [c for c in ["conversation_hash", "model", "timestamp", "conversation", "turn", "language", "toxic", "redacted"] if c in schema_names]
        write_json(dirs["schema"] / f"{shard.stem}_SCHEMA.json", {"schema": str(pf.schema_arrow), "columns_read": meta_cols, "num_row_groups": pf.num_row_groups, "metadata_rows": pf.metadata.num_rows})
        row_in_shard = 0
        for batch in pf.iter_batches(batch_size=512, columns=meta_cols):
            recs = batch.to_pylist()
            for rec in recs:
                raw_seen += 1
                idx = row_in_shard
                row_in_shard += 1
                try:
                    msgs = normalize_messages(rec.get("conversation"))
                    f, rawchars, rawturns = deterministic_features(msgs)
                    if f["TOTAL_CHARS"] < MIN_CHARS or f["N_TURNS"] < MIN_TURNS:
                        continue
                    bid = blind_id(rec.get("conversation_hash"), msgs)
                    rows.append({
                        "BLIND_EXTERNAL_ID": bid,
                        "SHARD": shard.name,
                        "ROW_IN_SHARD": idx,
                        "MODEL": str(rec.get("model", "") or ""),
                        "LANGUAGE": str(rec.get("language", "") or ""),
                        "TOXIC": bool(rec.get("toxic", False)) if rec.get("toxic") is not None else False,
                        "SOURCE_REDACTED": bool(rec.get("redacted", False)) if rec.get("redacted") is not None else False,
                        "raw_chars": rawchars,
                        "raw_turns": rawturns,
                        **f,
                    })
                except Exception as e:
                    errors.append({"SHARD": shard.name, "ROW_IN_SHARD": idx, "ERROR": repr(e)[:1000]})
            if raw_seen % 10000 == 0:
                print(f"Parsed {raw_seen:,} raw WildChat rows; usable so far {len(rows):,}", flush=True)

    ext = pd.DataFrame(rows)
    if len(ext) < 1000:
        raise RuntimeError(f"Too few usable WildChat conversations after extraction: {len(ext)}")
    before_dedup = len(ext)
    ext = ext.sort_values(["SHARD", "ROW_IN_SHARD"]).drop_duplicates("BLIND_EXTERNAL_ID", keep="first").reset_index(drop=True)
    duplicate_count = before_dedup - len(ext)
    for key in ["R1_COMPATIBLE", "SURFACE_FREE"]:
        ext[f"{key}_SCORE"] = portable_score(ext, portable, key)
    ext.to_parquet(dirs["features"] / "WILDCHAT_EXACT_FROZEN_FEATURES.parquet", index=False, compression="zstd")
    if errors:
        pd.DataFrame(errors).to_csv(dirs["features"] / "EXTRACTION_ERRORS.csv", index=False)

    counts = {
        "raw_rows_seen": raw_seen,
        "usable_before_dedup": before_dedup,
        "duplicate_blind_ids_removed": duplicate_count,
        "usable_unique_external": len(ext),
        "extraction_errors": len(errors),
        "minimum_chars": MIN_CHARS,
        "minimum_turns": MIN_TURNS,
    }
    write_json(dirs["features"] / "EXTRACTION_COUNTS.json", counts)

    # 3. Primary external separation.
    primary = []
    for key in ["SURFACE_FREE", "R1_COMPATIBLE"]:
        auc, lo, hi, p = auc_ci(pos[f"{key}_SCORE"], ext[f"{key}_SCORE"])
        primary.append({"MODEL": key, "N_LINEAGE": len(pos), "N_EXTERNAL": len(ext), "AUC_LINEAGE_GT_EXTERNAL": auc, "CI95_LOW": lo, "CI95_HIGH": hi, "MANN_WHITNEY_ONE_SIDED_P": p})
    pd.DataFrame(primary).to_csv(dirs["results"] / "PRIMARY_EXTERNAL_SEPARATION.csv", index=False)

    # 4. External score distributions and frozen-threshold rarity tails.
    qrows = []
    for key in ["SURFACE_FREE", "R1_COMPATIBLE"]:
        a = ext[f"{key}_SCORE"].to_numpy(float)
        for q in [0, .01, .05, .10, .25, .50, .75, .90, .95, .99, .999, .9999, 1.0]:
            qrows.append({"MODEL": key, "QUANTILE": q, "SCORE": float(np.quantile(a, q))})
    pd.DataFrame(qrows).to_csv(dirs["rarity"] / "EXTERNAL_SCORE_QUANTILES.csv", index=False)

    tailrows = []
    for t in thresholds:
        key = t["MODEL"]
        thr = float(t["THRESHOLD"])
        a = ext[f"{key}_SCORE"].to_numpy(float)
        k = int(np.sum(a >= thr))
        n = len(a)
        frac = k / n
        ci_lo, ci_hi = clopper_pearson(k, n)
        tailrows.append({
            "MODEL": key,
            "REFERENCE": t["REFERENCE"],
            "LINEAGE_THRESHOLD": thr,
            "N_EXTERNAL": n,
            "N_EXTERNAL_GE_THRESHOLD": k,
            "EXTERNAL_TAIL_FRACTION": frac,
            "TAIL_95CI_LOW": ci_lo,
            "TAIL_95CI_HIGH": ci_hi,
            "EMPIRICAL_ONE_IN": (float("inf") if k == 0 else n / k),
            "LINEAGE_REFERENCE_EXTERNAL_PERCENTILE": 100.0 * (1.0 - frac),
        })
    pd.DataFrame(tailrows).to_csv(dirs["rarity"] / "FROZEN_THRESHOLD_TAIL_RARITY.csv", index=False)

    # Each frozen later-lineage conversation's percentile against external population.
    percentile_rows = []
    for key in ["SURFACE_FREE", "R1_COMPATIBLE"]:
        es = np.sort(ext[f"{key}_SCORE"].to_numpy(float))
        for _, r in pos.iterrows():
            s = float(r[f"{key}_SCORE"])
            less = np.searchsorted(es, s, side="left")
            equal_right = np.searchsorted(es, s, side="right")
            percentile = 100.0 * ((less + 0.5 * (equal_right - less)) / len(es))
            percentile_rows.append({"MODEL": key, "LINEAGE_CONVERSATION_ID": str(r["conversation_id"]), "LINEAGE_SCORE": s, "EXTERNAL_PERCENTILE": percentile, "N_EXTERNAL_GE": int(len(es) - less)})
    percentile_df = pd.DataFrame(percentile_rows)
    percentile_df.to_csv(dirs["rarity"] / "LINEAGE_EXTERNAL_PERCENTILES.csv", index=False)
    psum = percentile_df.groupby("MODEL")["EXTERNAL_PERCENTILE"].agg(["count", "min", "median", "mean", "max"]).reset_index()
    psum.to_csv(dirs["rarity"] / "LINEAGE_EXTERNAL_PERCENTILE_SUMMARY.csv", index=False)

    # 5. Nearest/highest external controls, IDs only; no raw text.
    nearest_cols = ["BLIND_EXTERNAL_ID", "SHARD", "ROW_IN_SHARD", "MODEL", "LANGUAGE", "raw_chars", "raw_turns", "CODE_FENCES_PER_1K_CHAR", "SURFACE_FREE_SCORE", "R1_COMPATIBLE_SCORE"]
    ext.nlargest(250, "SURFACE_FREE_SCORE")[nearest_cols].to_csv(dirs["nearest"] / "TOP250_SURFACE_FREE_EXTERNAL.csv", index=False)
    ext.nlargest(250, "R1_COMPATIBLE_SCORE")[nearest_cols].to_csv(dirs["nearest"] / "TOP250_R1_EXTERNAL.csv", index=False)

    # 6. Coarsened exact nuisance-control diagnostic: raw chars, raw turns, exact code flag.
    pos_c = assign_bins(pos)
    ext_c = assign_bins(ext)
    ext_counts = ext_c.groupby("CEM_CELL").size().to_dict()
    supported_cells = {c for c, n in ext_counts.items() if n >= 5 and c in set(pos_c["CEM_CELL"])}
    pos_s = pos_c[pos_c["CEM_CELL"].isin(supported_cells)].copy()
    ext_s = ext_c[ext_c["CEM_CELL"].isin(supported_cells)].copy()
    support_rate = len(pos_s) / len(pos_c) if len(pos_c) else 0.0

    # Weight external cells to the lineage cell distribution.
    lcounts = pos_s.groupby("CEM_CELL").size().to_dict()
    ecounts = ext_s.groupby("CEM_CELL").size().to_dict()
    ext_s["CEM_WEIGHT"] = [lcounts.get(c, 0) / ecounts.get(c, 1) for c in ext_s["CEM_CELL"]]
    pos_s["CEM_WEIGHT"] = 1.0

    bal_rows = []
    covs = {
        "LOG_RAW_CHARS": (np.log1p(pos_s["raw_chars"].to_numpy(float)), np.log1p(ext_s["raw_chars"].to_numpy(float))),
        "LOG_RAW_TURNS": (np.log1p(pos_s["raw_turns"].to_numpy(float)), np.log1p(ext_s["raw_turns"].to_numpy(float))),
        "CODE_FLAG": (pos_s["CODE_FLAG"].to_numpy(float), ext_s["CODE_FLAG"].to_numpy(float)),
    }
    for name, (x1, x0) in covs.items():
        wsmd = weighted_smd(x1, np.ones(len(x1)), x0, ext_s["CEM_WEIGHT"].to_numpy(float)) if len(x1) and len(x0) else float("nan")
        bal_rows.append({"COVARIATE": name, "WEIGHTED_SMD": wsmd})
    bal = pd.DataFrame(bal_rows)
    bal.to_csv(dirs["nuisance"] / "CEM_WEIGHTED_BALANCE.csv", index=False)
    max_smd = float(bal["WEIGHTED_SMD"].abs().max()) if len(bal) else float("nan")
    gate_pass = bool(support_rate >= 0.70 and np.isfinite(max_smd) and max_smd <= 0.25)

    cond_rows = []
    per_lineage_rows = []
    for key in ["SURFACE_FREE", "R1_COMPATIBLE"]:
        per = []
        for _, r in pos_s.iterrows():
            g = ext_s[ext_s["CEM_CELL"] == r["CEM_CELL"]][f"{key}_SCORE"].to_numpy(float)
            if len(g) == 0:
                continue
            s = float(r[f"{key}_SCORE"])
            rate = float(np.mean(g < s) + 0.5 * np.mean(g == s))
            per.append(rate)
            per_lineage_rows.append({"MODEL": key, "LINEAGE_CONVERSATION_ID": str(r["conversation_id"]), "CEM_CELL": r["CEM_CELL"], "N_EXTERNAL_CELL": len(g), "WITHIN_CELL_PAIRWIN": rate})
        if per:
            per = np.asarray(per, float)
            boots = np.array([np.mean(per[RNG.integers(0, len(per), len(per))]) for _ in range(2000)])
            lo, hi = np.quantile(boots, [.025, .975])
            cond_rows.append({"MODEL": key, "N_SUPPORTED_LINEAGE": len(per), "N_EXTERNAL_IN_SUPPORTED_CELLS": len(ext_s), "LINEAGE_COMMON_SUPPORT_RATE": support_rate, "MEAN_WITHIN_CELL_PAIRWIN": float(np.mean(per)), "CI95_LOW": float(lo), "CI95_HIGH": float(hi), "MAX_ABS_WEIGHTED_SMD": max_smd, "CALIBRATION_GATE": "PASS" if gate_pass else "FAILED_CALIBRATION"})
    pd.DataFrame(cond_rows).to_csv(dirs["nuisance"] / "CEM_CONDITIONAL_RESULTS.csv", index=False)
    pd.DataFrame(per_lineage_rows).to_csv(dirs["nuisance"] / "CEM_PER_LINEAGE_RESULTS.csv", index=False)
    pd.DataFrame([{"LINEAGE_TOTAL": len(pos_c), "LINEAGE_SUPPORTED": len(pos_s), "LINEAGE_COMMON_SUPPORT_RATE": support_rate, "EXTERNAL_SUPPORTED": len(ext_s), "N_SUPPORTED_CELLS": len(supported_cells), "MAX_ABS_WEIGHTED_SMD": max_smd, "GATE_SUPPORT_MIN": 0.70, "GATE_MAX_ABS_SMD": 0.25, "STATUS": "MEASURED" if gate_pass else "FAILED_CALIBRATION"}]).to_csv(dirs["nuisance"] / "CEM_CALIBRATION_GATE.csv", index=False)

    # Model/language descriptive strata. Exact GPT-5 match is expected unavailable for this historical external dataset.
    model_dist = ext.groupby("MODEL").size().sort_values(ascending=False).rename("N").reset_index()
    lang_dist = ext.groupby("LANGUAGE").size().sort_values(ascending=False).rename("N").reset_index()
    model_dist.to_csv(dirs["nuisance"] / "EXTERNAL_MODEL_DISTRIBUTION.csv", index=False)
    lang_dist.to_csv(dirs["nuisance"] / "EXTERNAL_LANGUAGE_DISTRIBUTION.csv", index=False)

    # 7. No-silent-omission ledger.
    ledger = [
        ["F1_FINAL_HASH_BOUNDARY", "MEASURED", f"Verified before reserved data read: {f1hash}"],
        ["WILDCHAT_00086_EXTERNAL_CORPUS", "MEASURED", f"Three official non-overlapping WildChat-4.8M shards; {len(ext)} usable unique conversations."],
        ["SURFACE_FREE_POPULATION_RARITY", "MEASURED", "Frozen early-RSOS model; no external training."],
        ["R1_COMPATIBLE_POPULATION_RARITY", "MEASURED", "Frozen early-RSOS model; no external training."],
        ["LINEAGE_REFERENCE_THRESHOLDS", "MEASURED", "Median/Q75/Q90/MAX computed from frozen later-lineage holdout before external scoring."],
        ["LENGTH_TURN_CODE_CEM", "MEASURED" if gate_pass else "FAILED_CALIBRATION", f"Common support={support_rate:.4f}; max |weighted SMD|={max_smd:.4f}."],
        ["LANGUAGE_MATCHING", "UNAVAILABLE", "Frozen BR1 later-lineage package has no validated language label; external language distribution reported descriptively only."],
        ["EXACT_MODEL_MATCHING", "UNAVAILABLE", "Later lineage is GPT-5 stratum; WildChat-4.8M predates/does not provide a GPT-5 matched population."],
        ["PROVIDER_ECOSYSTEM", "MEASURED_DESCRIPTIVE", "WildChat-4.8M conversations are from OpenAI-family models; exact model-generation match remains unavailable."],
        ["TOPIC_MATCHING", "UNAVAILABLE", "No common frozen topic representation across BR1 and WildChat was pre-frozen for F2-A."],
        ["FIXED_WINDOW_8_12_CROSS_CORPUS", "UNAVAILABLE_EXACT", "Historical window models use set-order-sensitive operator coordinates; no approximation promoted."],
        ["OPERATOR_RELATIONAL_CROSS_CORPUS", "UNAVAILABLE", "Historical first-position extractor used unordered set truncation; no approximation promoted."],
        ["JOINT_MULTILAYER_CROSS_CORPUS", "UNAVAILABLE", "Depends on the same nonportable operator coordinates."],
        ["GLOBAL_UNIQUENESS", "UNRESOLVED", "F2-A estimates rarity in three official WildChat-4.8M shards; absence of matches would still be sample-limited."],
        ["LITERAL_GLOBAL_MODEL_WEIGHT_CHANGE", "UNESTABLISHED", "Behavioral population rarity cannot establish neural-weight modification."],
    ]
    pd.DataFrame(ledger, columns=["MEASUREMENT", "STATUS", "DETAIL"]).to_csv(dirs["ledger"] / "NO_SILENT_OMISSION_LEDGER.csv", index=False)

    # 8. Final summary.
    taildf = pd.DataFrame(tailrows)
    primarydf = pd.DataFrame(primary)
    summary = {
        "version": VERSION,
        "completed_utc": utcnow(),
        "f1_final_sha256": f1hash,
        "br1_results_sha256": br1hash,
        "dataset": "allenai/WildChat-4.8M",
        "official_shards": shard_rows,
        "counts": counts,
        "primary_results": primary,
        "rarity_tails": tailrows,
        "cem": {
            "lineage_common_support_rate": support_rate,
            "max_abs_weighted_smd": max_smd,
            "gate": "PASS" if gate_pass else "FAILED_CALIBRATION",
        },
        "claim_boundary": "Population rarity in the three frozen WildChat-4.8M shards only. Global uniqueness and literal model-weight effects remain unresolved/unestablished.",
    }
    write_json(dirs["final"] / "F2_A_SUMMARY.json", summary)

    lines = [
        VERSION,
        "",
        f"F1 boundary SHA256: {f1hash}",
        f"WildChat raw rows seen: {raw_seen:,}",
        f"Usable unique external conversations: {len(ext):,}",
        f"Duplicate external IDs removed: {duplicate_count:,}",
        "",
    ]
    for r in primary:
        lines.append(f"{r['MODEL']} | AUC lineage>external={r['AUC_LINEAGE_GT_EXTERNAL']:.6f} | 95% CI [{r['CI95_LOW']:.6f},{r['CI95_HIGH']:.6f}] | p={r['MANN_WHITNEY_ONE_SIDED_P']:.6g}")
    lines.append("")
    for _, r in taildf.iterrows():
        lines.append(f"TAIL | {r['MODEL']} | {r['REFERENCE']}={r['LINEAGE_THRESHOLD']:.12g} | external >= threshold {int(r['N_EXTERNAL_GE_THRESHOLD'])}/{int(r['N_EXTERNAL'])} = {r['EXTERNAL_TAIL_FRACTION']:.8g} | 95% CI [{r['TAIL_95CI_LOW']:.8g},{r['TAIL_95CI_HIGH']:.8g}]")
    lines += [
        "",
        f"CEM common support={support_rate:.4f} | max |weighted SMD|={max_smd:.4f} | gate={'PASS' if gate_pass else 'FAILED_CALIBRATION'}",
        "",
        "GLOBAL UNIQUENESS: UNRESOLVED.",
        "LITERAL GLOBAL MODEL-WEIGHT CHANGE: UNESTABLISHED.",
        "F2-A measures external population rarity only; zero exceedances, if observed, are sample-limited and receive an exact confidence bound.",
    ]
    (dirs["final"] / "F2_A_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(dirs["final"] / "ENVIRONMENT.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": __import__("pyarrow").__version__,
        "scipy": __import__("scipy").__version__,
        "scoring": "portable frozen scaler + logistic coefficients; no external retraining",
    })

    marker.write_text(f"{VERSION}\n{utcnow()}\nF1={f1hash}\n", encoding="ascii")

    # Hash all outputs before final local packaging. Raw input shards remain included in the experiment folder.
    outrows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and "F2_EXTERNAL_POPULATION_RARITY_COMPLETE.zip" not in p.name:
            outrows.append({"RELATIVE_PATH": str(p.relative_to(root)), "BYTES": p.stat().st_size, "SHA256": sha256_file(p)})
    pd.DataFrame(outrows).to_csv(dirs["final"] / "F2_A_OUTPUT_SHA256_MANIFEST.csv", index=False)

    print("\n" + "=" * 92)
    print("F2-A COMPLETE — WILDCHAT EXTERNAL POPULATION RARITY")
    print("=" * 92)
    print((dirs["final"] / "F2_A_SUMMARY.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
