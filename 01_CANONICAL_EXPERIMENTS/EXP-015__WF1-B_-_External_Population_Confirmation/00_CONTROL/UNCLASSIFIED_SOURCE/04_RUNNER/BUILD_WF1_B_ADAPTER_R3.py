from pathlib import Path

orig = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY\WF1_B_EXTERNAL_POPULATION_PARQUET\04_RUNNER\F2_B_FROZEN_PASSIVE_BASELINE\RUN_F2_B.py")
out  = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY\WF1_B_EXTERNAL_POPULATION_PARQUET\04_RUNNER\F2_B_FROZEN_PASSIVE_BASELINE\RUN_WF1_B_FRESH14_ADAPTER_R3.py")

sha = "8538243cf3cf6513c46658d4dc2da19f6bde729f3e04197abe9abac30aae8bfe"

text = orig.read_text(encoding="utf-8")

marker = "    # Fresh file preflight by filename+size only, and hard exclusion of prior shards."

insert = f"""    # WF1-B shard-selection adapter only.
    wf1b_name='train-00014-of-00086.parquet'
    wf1b_expected_sha='{sha}'
    wf1b_path=inp/wf1b_name

    if not wf1b_path.exists():
        raise Abort(f'WF1-B frozen shard missing: {{wf1b_path}}')

    wf1b_hash=hashlib.sha256()
    with open(wf1b_path,'rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            wf1b_hash.update(chunk)

    if wf1b_hash.hexdigest()!=wf1b_expected_sha:
        raise Abort(f'WF1-B shard SHA256 mismatch: {{wf1b_hash.hexdigest()}} != {{wf1b_expected_sha}}')

    SPEC['fresh_shards']={{wf1b_name: wf1b_path.stat().st_size}}

{marker}"""

if marker not in text:
    raise RuntimeError("Patch marker not found")

text = text.replace(marker, insert, 1)

text = text.replace(
    "Only shards 07,22,41,51,52,54; 49/80/85 excluded.",
    "WF1-B frozen fresh shard 14 only; prior F2 shards excluded."
)

text = text.replace(
    "This experiment estimates prevalence only in the six frozen fresh shards.",
    "This experiment estimates prevalence only in frozen WF1-B fresh shard 14."
)

text = text.replace(
    "Population rarity in six fresh frozen WildChat shards.",
    "Population rarity in frozen WF1-B fresh WildChat shard 14."
)

out.write_text(text, encoding="utf-8", newline="\n")

print(out)
