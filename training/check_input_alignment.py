
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = root / "data" / "processed"
patient_id = "10002013"
target_id = "899e40a8-0562ad87-0f3253ea-4d376ad5-ad830fc2"

source = pd.read_csv(data / "multimodal" / "test_metadata.csv")
source["cxr_time"] = pd.to_datetime(source["cxr_time"])
source["embedding_index"] = np.arange(len(source))

patient = source[
    source["subject_id"].astype(str) == patient_id
].sort_values(["cxr_time", "dicom_id"]).reset_index(drop=True)

sequence_meta = pd.read_csv(data / "sequences" / "test_metadata.csv")
matches = sequence_meta.index[
    sequence_meta["target_dicom_id"].astype(str) == target_id
].tolist()

if len(matches) != 1:
    raise ValueError(f"Expected one target match, found {len(matches)}")

row = matches[0]
X = np.load(data / "sequences" / "test_X.npy", mmap_mode="r")
embeddings = np.load(
    data / "multimodal" / "test_cxr_embeddings.npy",
    mmap_mode="r"
)

print("Sequence row:", row)
print("\nChronological source visits:")
print(patient[["dicom_id", "cxr_time"]].to_string(index=False))

if len(patient) < 4:
    raise ValueError("Patient has fewer than four visits")

print("\nInput embedding comparisons:")
for j in range(3):
    source_index = int(patient.iloc[j]["embedding_index"])
    difference = np.max(
        np.abs(X[row, j, :512] - embeddings[source_index])
    )
    print(
        f"Visit {j + 1}: "
        f"{patient.iloc[j]['dicom_id']} | "
        f"max absolute difference: {difference:.10f}"
    )

print("\nExpected target:", patient.iloc[3]["dicom_id"])
print("Saved target:", sequence_meta.iloc[row]["target_dicom_id"])
