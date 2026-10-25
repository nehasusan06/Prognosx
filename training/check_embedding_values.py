
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "processed" / "sequences"
SRC = ROOT / "data" / "processed" / "multimodal"

source = np.load(SRC / "test_cxr_embeddings.npy", mmap_mode="r")
source_meta = pd.read_csv(SRC / "test_metadata.csv")
X = np.load(SEQ / "test_X.npy", mmap_mode="r")
meta = pd.read_csv(SEQ / "test_metadata.csv")

source_meta = source_meta.copy()
source_meta["_idx"] = np.arange(len(source_meta))
source_meta["_time"] = pd.to_datetime(source_meta["cxr_time"])
source_meta["_subject"] = source_meta["subject_id"].astype(str).str.strip()
source_meta["_dicom"] = source_meta["dicom_id"].astype(str).str.strip()

ordered = source_meta.sort_values(
    ["subject_id", "_time"]
).reset_index(drop=True)

print("SOURCE EMBEDDING NORMS")
source_norms = np.linalg.norm(np.asarray(source, dtype=np.float32), axis=1)
print("Min:", source_norms.min())
print("Median:", np.median(source_norms))
print("Max:", source_norms.max())

print("\nSAVED INPUT EMBEDDING NORMS")
saved_norms = np.linalg.norm(
    np.asarray(X[:, :, :512], dtype=np.float32), axis=2
)
print("Min:", saved_norms.min())
print("Median:", np.median(saved_norms))
print("Max:", saved_norms.max())

# Reconstruct source sequences using the timestamp ordering.
expected = {}
for subject, group in ordered.groupby("_subject", sort=False):
    indices = group.index.to_list()
    for i in range(max(0, len(indices) - 3)):
        inputs = indices[i:i + 3]
        target = indices[i + 3]
        key = (subject, ordered.loc[target, "_dicom"])
        expected[key] = inputs

print("\nCOMPARISON FOR FIRST MISMATCH EXAMPLE")
checked = 0

for row in range(len(meta)):
    subject = str(meta.iloc[row]["subject_id"]).strip()
    target_id = str(meta.iloc[row]["target_dicom_id"]).strip()
    inputs = expected.get((subject, target_id))

    if inputs is None:
        continue

    for pos, sorted_idx in enumerate(inputs):
        original_idx = int(ordered.loc[sorted_idx, "_idx"])
        raw = np.array(source[original_idx], dtype=np.float32, copy=True)
        saved = np.array(X[row, pos, :512], dtype=np.float32, copy=True)

        raw_norm = raw / max(np.linalg.norm(raw), 1e-12)
        raw_diff = np.max(np.abs(saved - raw))
        normalized_diff = np.max(np.abs(saved - raw_norm))

        if raw_diff > 1e-6 or normalized_diff > 1e-6:
            print("Sequence row:", row, "input visit:", pos + 1)
            print("Source DICOM:", ordered.loc[sorted_idx, "_dicom"])
            print("Saved input norm:", float(np.linalg.norm(saved)))
            print("Raw source norm:", float(np.linalg.norm(raw)))
            print("Max difference vs raw:", float(raw_diff))
            print("Max difference vs normalized:", float(normalized_diff))
            checked += 1
            if checked >= 5:
                break

    if checked >= 5:
        break

print("\nDiagnostic complete. No files modified.")
