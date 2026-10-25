
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = root / "data" / "processed"
multi = data / "multimodal"
seq = data / "sequences"

source_meta = pd.read_csv(multi / "test_metadata.csv")
sequence_meta = pd.read_csv(seq / "test_metadata.csv")

source_embeddings = np.load(
    multi / "test_cxr_embeddings.npy", mmap_mode="r"
)
X = np.load(seq / "test_X.npy", mmap_mode="r")
y = np.load(seq / "test_y.npy", mmap_mode="r")

assert len(source_meta) == len(source_embeddings)
assert len(sequence_meta) == len(X) == len(y)

source_meta["embedding_index"] = np.arange(len(source_meta))
source_meta["subject_id"] = source_meta["subject_id"].astype(str)
source_meta["dicom_id"] = source_meta["dicom_id"].astype(str)

# DICOM IDs should identify one source embedding each.
if source_meta["dicom_id"].duplicated().any():
    raise ValueError("Duplicate source DICOM IDs found.")

source_index = dict(zip(
    source_meta["dicom_id"],
    source_meta["embedding_index"]
))

input_errors = []
target_errors = []
missing_ids = []

for i, row in sequence_meta.iterrows():
    patient_id = str(row["subject_id"])

    # Find this patient's source visits and apply the same tie-break rule
    # as the sequence builder: cxr_time, then dicom_id.
    patient = source_meta[
        source_meta["subject_id"] == patient_id
    ].copy()
    patient["cxr_time"] = pd.to_datetime(
        patient["cxr_time"], errors="coerce"
    )
    patient = patient.sort_values(
        ["cxr_time", "dicom_id"]
    ).reset_index(drop=True)

    target_id = str(row["target_dicom_id"])
    target_positions = patient.index[
        patient["dicom_id"] == target_id
    ].tolist()

    if len(target_positions) != 1:
        missing_ids.append((i, target_id))
        continue

    target_pos = target_positions[0]

    if target_pos < 3:
        missing_ids.append((i, target_id))
        continue

    expected_inputs = patient.iloc[target_pos - 3:target_pos]
    actual_ids = expected_inputs["dicom_id"].tolist()

    # Compare the three input image embeddings.
    for j, dicom_id in enumerate(actual_ids):
        k = source_index.get(dicom_id)
        if k is None:
            missing_ids.append((i, dicom_id))
            continue

        diff = float(np.max(np.abs(
            X[i, j, :512] - source_embeddings[k]
        )))
        if diff > 1e-6:
            input_errors.append((i, j, dicom_id, diff))

    # Compare saved target embedding with target source embedding.
    k = source_index.get(target_id)
    if k is None:
        missing_ids.append((i, target_id))
    else:
        diff = float(np.max(np.abs(y[i] - source_embeddings[k])))
        if diff > 1e-6:
            target_errors.append((i, target_id, diff))

print("Test sequences:", len(sequence_meta))
print("Input embedding mismatches:", len(input_errors))
print("Target embedding mismatches:", len(target_errors))
print("Missing/invalid mappings:", len(missing_ids))

if input_errors:
    print("First input mismatches:", input_errors[:5])
if target_errors:
    print("First target mismatches:", target_errors[:5])
if missing_ids:
    print("First missing mappings:", missing_ids[:5])

if not input_errors and not target_errors and not missing_ids:
    print("PASS: All embeddings match under the stated chronological ordering.")
else:
    print("FAIL: Investigate the listed rows before reporting final metrics.")
