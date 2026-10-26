from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "multimodal"
SEQ = ROOT / "data" / "processed" / "sequences"

source = np.load(SRC / "test_cxr_embeddings.npy").astype(np.float32)
source_meta = pd.read_csv(SRC / "test_metadata.csv")
X = np.load(SEQ / "test_X.npy", mmap_mode="r")
seq_meta = pd.read_csv(SEQ / "test_metadata.csv")

row = 160
subject = str(seq_meta.iloc[row]["subject_id"]).strip()

print("Sequence row:", row)
print(seq_meta.iloc[row].to_string())

mask = source_meta["subject_id"].astype(str).str.strip() == subject
indices = np.flatnonzero(mask.to_numpy())
patient_embeddings = source[indices]
patient_norms = np.linalg.norm(patient_embeddings, axis=1)

for pos in range(3):
    saved = np.asarray(X[row, pos, :512], dtype=np.float32)
    saved_norm = np.linalg.norm(saved)

    scores = (patient_embeddings @ saved) / np.maximum(
        patient_norms * saved_norm, 1e-12
    )

    best = int(np.argmax(scores))
    source_idx = int(indices[best])
    source_row = source_meta.iloc[source_idx]

    print(f"\nINPUT VISIT {pos + 1}")
    print("Source index:", source_idx)
    print("DICOM:", source_row["dicom_id"])
    print("Timestamp:", source_row["cxr_time"])
    print("Cosine similarity:", float(scores[best]))
    print(
        "Max absolute difference:",
        float(np.max(np.abs(saved - source[source_idx])))
    )

print("\nDone. No files modified.")