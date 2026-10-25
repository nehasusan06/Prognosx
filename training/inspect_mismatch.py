
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = root / "data" / "processed"
multi = data / "multimodal"
seq = data / "sequences"

row = 8

source = pd.read_csv(multi / "test_metadata.csv")
source["cxr_time"] = pd.to_datetime(
    source["cxr_time"], errors="coerce"
)
source["embedding_index"] = np.arange(len(source))
source["subject_id"] = source["subject_id"].astype(str)
source["dicom_id"] = source["dicom_id"].astype(str)

meta = pd.read_csv(seq / "test_metadata.csv")
meta["subject_id"] = meta["subject_id"].astype(str)

X = np.load(seq / "test_X.npy", mmap_mode="r")
y = np.load(seq / "test_y.npy", mmap_mode="r")
embeddings = np.load(
    multi / "test_cxr_embeddings.npy", mmap_mode="r"
)

record = meta.iloc[row]
patient_id = record["subject_id"]
target_id = str(record["target_dicom_id"])

patient = source[
    source["subject_id"] == patient_id
].sort_values("cxr_time").reset_index(drop=True)

target_positions = patient.index[
    patient["dicom_id"] == target_id
].tolist()

print("Sequence row:", row)
print("Patient:", patient_id)
print("Target ID:", target_id)
print("\nSaved sequence metadata:")
print(record.to_string())

if len(target_positions) != 1:
    print("\nTarget occurs", len(target_positions), "times in source metadata.")
else:
    target_pos = target_positions[0]
    expected = patient.iloc[max(0, target_pos - 3):target_pos]

    print("\nThree preceding source visits:")
    print(expected[["dicom_id", "cxr_time"]].to_string(index=False))

    print("\nInput embedding differences:")
    for j, (_, visit) in enumerate(expected.iterrows()):
        source_idx = int(visit["embedding_index"])
        diff = np.max(
            np.abs(X[row, j, :512] - embeddings[source_idx])
        )
        print(
            f"Input {j + 1}: {visit['dicom_id']} | "
            f"max difference = {diff:.10f}"
        )

print("\nSaved target embedding difference:")
target_rows = source.index[source["dicom_id"] == target_id].tolist()
if len(target_rows) == 1:
    idx = int(source.loc[target_rows[0], "embedding_index"])
    print(float(np.max(np.abs(y[row] - embeddings[idx]))))
else:
    print("Target source mapping is not unique.")
