
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "processed" / "sequences" / "test_metadata.csv"

df = pd.read_csv(path)

for col in ["target_cxr_time", "sequence_start", "sequence_end"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

bad = df[df["target_cxr_time"] <= df["sequence_end"]].copy()

print("Total test rows:", len(df))
print("Inconsistent rows:", len(bad))
print("\nFirst 20 inconsistent rows:")
print(bad.head(20).to_string(index=False))

print("\nInconsistent rows by patient:")
print(bad["subject_id"].value_counts().head(15).to_string())

print("\nDifference between target and sequence end (minutes):")
print(
    (
        bad["target_cxr_time"] - bad["sequence_end"]
    ).dt.total_seconds().div(60).describe()
)
