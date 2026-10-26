from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "multimodal"
SEQ = ROOT / "data" / "processed" / "sequences"

source = np.load(SRC / "test_cxr_embeddings.npy", mmap_mode="r")
source_meta = pd.read_csv(SRC / "test_metadata.csv")
X = np.load(SEQ / "test_X.npy", mmap_mode="r")
y = np.load(SEQ / "test_y.npy", mmap_mode="r")
seq_meta = pd.read_csv(SEQ / "test_metadata.csv")

# Preserve the original source-array indices.
meta = source_meta.copy()
meta["_idx"] = np.arange(len(meta))
meta["_subject"] = meta["subject_id"].astype(str).str.strip()
meta["_dicom"] = meta["dicom_id"].astype(str).str.strip()
meta["_time"] = pd.to_datetime(meta["cxr_time"], errors="coerce")

# Use a deterministic order for ties.
ordered = meta.sort_values(
    ["_subject", "_time", "_dicom"],
    kind="mergesort"
).reset_index(drop=True)

# Map each patient and target DICOM to its expected three input DICOMs.
expected = {}
for subject, group in ordered.groupby("_subject", sort=False):
    indices = group.index.to_list()

    for i in range(len(indices) - 3):
        inputs = indices[i:i + 3]
        target = indices[i + 3]
        key = (subject, ordered.loc[target, "_dicom"])
        expected.setdefault(key, []).append(inputs)

matched_rows = 0
unmatched_rows = 0
ambiguous_rows = 0
input_mismatches = 0
target_mismatches = 0
timestamp_ties = 0

for row in range(len(seq_meta)):
    subject = str(seq_meta.iloc[row]["subject_id"]).strip()
    target_dicom = str(seq_meta.iloc[row]["target_dicom_id"]).strip()

    candidates = expected.get((subject, target_dicom), [])

    if not candidates:
        unmatched_rows += 1
        continue

    if len(candidates) != 1:
        ambiguous_rows += 1
        continue

    matched_rows += 1
    inputs = candidates[0]

    for pos, sorted_idx in enumerate(inputs):
        original_idx = int(ordered.loc[sorted_idx, "_idx"])
        expected_vec = np.asarray(source[original_idx], dtype=np.float32)
        saved_vec = np.asarray(X[row, pos, :512], dtype=np.float32)

        if not np.allclose(saved_vec, expected_vec, atol=1e-6, rtol=1e-5):
            input_mismatches += 1

    target_idx = int(
        ordered.loc[inputs[-1] + 1, "_idx"]
    )
    target_vec = np.asarray(source[target_idx], dtype=np.float32)
    saved_target = np.asarray(y[row], dtype=np.float32)

    if not np.allclose(saved_target, target_vec, atol=1e-6, rtol=1e-5):
        target_mismatches += 1

    # Count cases where any input and the target share a timestamp.
    target_time = ordered.loc[inputs[-1] + 1, "_time"]
    input_times = [ordered.loc[idx, "_time"] for idx in inputs]

    if any(t == target_time for t in input_times):
        timestamp_ties += 1

print("\n=== ALIGNMENT AUDIT ===")
print("Saved test sequences:", len(seq_meta))
print("Rows matched:", matched_rows)
print("Unmatched rows:", unmatched_rows)
print("Ambiguous target mappings:", ambiguous_rows)
print("Input embedding mismatches:", input_mismatches)
print("Target embedding mismatches:", target_mismatches)
print("Input-target timestamp ties:", timestamp_ties)
print("\nAudit finished. No files modified.")