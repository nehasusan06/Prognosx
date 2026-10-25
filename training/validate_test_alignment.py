
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "data" / "processed" / "sequences"

X = np.load(SEQ_DIR / "test_X.npy", mmap_mode="r")
y = np.load(SEQ_DIR / "test_y.npy", mmap_mode="r")
meta = pd.read_csv(SEQ_DIR / "test_metadata.csv")

print("=== Shape and row alignment ===")
print("X:", X.shape)
print("y:", y.shape)
print("Metadata:", meta.shape)

assert len(X) == len(y) == len(meta), (
    "ERROR: X, y and metadata row counts do not match"
)
print("PASS: All three files have matching row counts.")

print("\n=== Metadata checks ===")
meta["target_cxr_time"] = pd.to_datetime(
    meta["target_cxr_time"], errors="coerce"
)
meta["sequence_start"] = pd.to_datetime(
    meta["sequence_start"], errors="coerce"
)
meta["sequence_end"] = pd.to_datetime(
    meta["sequence_end"], errors="coerce"
)

print("Missing target times:", meta["target_cxr_time"].isna().sum())
print("Targets not after sequence end:",
      (meta["target_cxr_time"] <= meta["sequence_end"]).sum())
print("Sequence end before sequence start:",
      (meta["sequence_end"] < meta["sequence_start"]).sum())
print("Duplicate target IDs:", meta["target_dicom_id"].duplicated().sum())

print("\n=== Patient-level checks ===")
print("Unique patients:", meta["subject_id"].nunique())
print("Rows per patient (top 10):")
print(meta["subject_id"].value_counts().head(10).to_string())

print("\n=== Conclusion ===")
print(
    "These checks validate file sizes and basic metadata consistency. "
    "To prove that every target embedding matches its DICOM ID and that "
    "the three inputs are the immediately preceding visits, we must also "
    "compare the sequence builder's source rows and embedding indices."
)
