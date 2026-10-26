from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "multimodal"
SEQ = ROOT / "data" / "processed" / "sequences"

embeddings = np.load(
    SRC / "test_cxr_embeddings.npy",
    mmap_mode="r"
)
metadata = pd.read_csv(SRC / "test_metadata.csv")

X = np.load(SEQ / "test_X.npy", mmap_mode="r")
y = np.load(SEQ / "test_y.npy", mmap_mode="r")
saved_meta = pd.read_csv(SEQ / "test_metadata.csv")

# Reproduce the original sequence builder's ordering steps.
metadata["cxr_time"] = pd.to_datetime(
    metadata["cxr_time"], errors="coerce"
)
metadata["embedding_index"] = np.arange(len(metadata))

metadata = metadata.sort_values(
    ["subject_id", "cxr_time"]
).reset_index(drop=True)

expected_inputs = []
expected_targets = []
expected_rows = []

for subject_id, patient_data in metadata.groupby("subject_id"):
    patient_data = patient_data.sort_values("cxr_time")
    indices = patient_data["embedding_index"].astype(int).tolist()

    if len(indices) <= 3:
        continue

    for i in range(len(indices) - 3):
        input_indices = indices[i:i + 3]
        target_index = indices[i + 3]

        target_row = patient_data.iloc[i + 3]

        expected_inputs.append(input_indices)
        expected_targets.append(target_index)
        expected_rows.append({
            "subject_id": subject_id,
            "target_dicom_id": str(target_row["dicom_id"]).strip(),
            "target_cxr_time": target_row["cxr_time"],
        })

expected_meta = pd.DataFrame(expected_rows)

print("Reconstructed sequences:", len(expected_inputs))
print("Saved sequences:", len(X))

if len(expected_inputs) != len(X):
    print(
        "\nWARNING: sequence counts differ. "
        "Do not compare row positions directly."
    )
    raise SystemExit(1)

input_mismatch_count = 0
input_vector_checks = 0
target_mismatch_count = 0
target_vector_checks = 0

for row, indices in enumerate(expected_inputs):
    for pos, source_idx in enumerate(indices):
        expected_vector = np.asarray(
            embeddings[source_idx], dtype=np.float32
        )
        saved_vector = np.asarray(
            X[row, pos, :512], dtype=np.float32
        )
        input_vector_checks += 1

        if not np.allclose(
            saved_vector, expected_vector,
            atol=1e-6, rtol=1e-5
        ):
            input_mismatch_count += 1

    target_idx = expected_targets[row]
    expected_target = np.asarray(
        embeddings[target_idx], dtype=np.float32
    )
    saved_target = np.asarray(y[row], dtype=np.float32)
    target_vector_checks += 1

    if not np.allclose(
        saved_target, expected_target,
        atol=1e-6, rtol=1e-5
    ):
        target_mismatch_count += 1

print("\n=== ORIGINAL-ORDER REPRODUCTION ===")
print("Input vector comparisons:", input_vector_checks)
print("Input vector mismatches:", input_mismatch_count)
print("Target vector comparisons:", target_vector_checks)
print("Target vector mismatches:", target_mismatch_count)

same_target_id = 0
for row in range(min(len(expected_meta), len(saved_meta))):
    a = str(expected_meta.iloc[row]["target_dicom_id"]).strip()
    b = str(saved_meta.iloc[row]["target_dicom_id"]).strip()
    if a == b:
        same_target_id += 1

print(
    "Target DICOM IDs matching at the same row:",
    same_target_id,
    "of",
    min(len(expected_meta), len(saved_meta))
)

print("\nNo files modified.")