from pathlib import Path
import pyarrow.parquet as pq

root = Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY\WF1_B_EXTERNAL_POPULATION_PARQUET")
raw = root / "02_DATASETS" / "RAW_PARQUET"

files = sorted(raw.glob("*.parquet"))

print("WF1-B WILDCHAT SCHEMA INSPECTION")
print("=" * 70)
print("SHARDS:", len(files))

for p in files:
    pf = pq.ParquetFile(p)
    print()
    print("FILE:", p.name)
    print("ROWS:", pf.metadata.num_rows)
    print("ROW_GROUPS:", pf.metadata.num_row_groups)
    print("SCHEMA:")
    print(pf.schema_arrow)
    print("-" * 70)
