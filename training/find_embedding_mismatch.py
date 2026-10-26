
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

SRC = ROOT / "data" / "processed" / "multimodal"
SEQ = ROOT / "data" / "processed" / "sequences"

source_path = SRC / "test_cxr_embeddings.npy"
source_meta_path = SRC / "test_metadata.csv"
sequence_path = SEQ / "test_X.npy"
sequence_meta_path = SEQ / "test_metadata.csv"

# --------------------------------------------------
# 2. Load source embeddings and saved sequences
# --------------------------------------------------
print("Loading files...")

source = np.load(source_path).astype(np.float32)
source_meta = pd.read_csv(source_meta_path)

X = np.load(sequence_path, mmap_mode="r")
seq_meta = pd.read_csv(sequence_meta_path)

print("Source embeddings shape:", source.shape)
print("Source metadata shape:", source_meta.shape)
print("Saved sequence shape:", X.shape)
print("Sequence metadata shape:", seq_meta.shape)

# --------------------------------------------------
# 3. Prepare metadata
# --------------------------------------------------
sm = source_meta.copy()

sm["_original_idx"] = np.arange(len(sm))
sm["_subject"] = sm["subject_id"].astype(str).str.strip()
sm["_time"] = pd.to_datetime(sm["cxr_time"], errors="coerce")
sm["_dicom"] = sm["dicom_id"].astype(str).str.strip()

# Match the ordering used by the current sequence validator.
ordered = sm.sort_values(
    ["subject_id", "_time"]
).reset_index(drop=True)

# --------------------------------------------------
# 4. Reconstruct expected three-visit sequences
# --------------------------------------------------
expected = {}

for subject, group in ordered.groupby("_subject", sort=False):
    indices = group.index.to_list()

    for i in range(len(indices) - 3):
        input_indices = indices[i:i + 3]
        target_idx = indices[i + 3]

        target_dicom = ordered.loc[target_idx, "_dicom"]
        key = (subject, target_dicom)

        expected.setdefault(key, []).append(input_indices)

# --------------------------------------------------
# 5. Find the first mismatching sequence
# --------------------------------------------------
print("\nSearching for a sequence to inspect...")

found = False

for row in range(len(seq_meta)):
    subject = str(seq_meta.iloc[row]["subject_id"]).strip()
    target_dicom = str(
        seq_meta.iloc[row]["target_dicom_id"]
    ).strip()

    candidates = expected.get((subject, target_dicom), [])

    if len(candidates) != 1:
        continue

    input_indices = candidates[0]
    mismatches = []

    for pos, sorted_idx in enumerate(input_indices):
        original_idx = int(
            ordered.loc[sorted_idx, "_original_idx"]
        )

        saved = np.asarray(
            X[row, pos, :512], dtype=np.float32
        )
        expected_embedding = source[original_idx]

        difference = float(
            np.max(np.abs(saved - expected_embedding))
        )

        if difference > 1e-6:
            mismatches.append((pos, sorted_idx, difference))

    if mismatches:
        print("\nSequence row:", row)
        print("Patient:", subject)
        print("Target DICOM:", target_dicom)
        print("Number of input mismatches:", len(mismatches))

        # Inspect the first mismatching input visit.
        pos, sorted_idx, expected_diff = mismatches[0]

        original_idx = int(
            ordered.loc[sorted_idx, "_original_idx"]
        )

        saved = np.asarray(
            X[row, pos, :512], dtype=np.float32
        )

        print("\nFirst mismatching input visit:", pos + 1)
        print(
            "Expected DICOM:",
            ordered.loc[sorted_idx, "_dicom"]
        )
        print(
            "Expected timestamp:",
            ordered.loc[sorted_idx, "_time"]
        )
        print("Expected source index:", original_idx)
        print(
            "Difference from expected embedding:",
            expected_diff
        )

        # --------------------------------------------------
        # 6. Search all source embeddings for the closest match
        # --------------------------------------------------
        source_norms = np.linalg.norm(source, axis=1)
        saved_norm = np.linalg.norm(saved)

        cosine_scores = (source @ saved) / np.maximum(
            source_norms * saved_norm, 1e-12
        )

        # Print the five closest candidates.
        top_indices = np.argsort(cosine_scores)[-5:][::-1]

        print("\nFive closest source embeddings:")

        for rank, idx in enumerate(top_indices, start=1):
            source_row = source_meta.iloc[int(idx)]

            max_diff = float(
                np.max(np.abs(saved - source[int(idx)]))
            )

            print(f"\nCandidate {rank}")
            print("Source index:", int(idx))
            print("Patient:", source_row["subject_id"])
            print("DICOM:", source_row["dicom_id"])
            print("Timestamp:", source_row["cxr_time"])
            print(
                "Cosine similarity:",
                float(cosine_scores[int(idx)])
            )
            print("Maximum absolute difference:", max_diff)

        found = True
        break

if not found:
    print(
        "\nNo mismatching sequence was found using this "
        "reconstructed mapping, or the mapping was ambiguous."
    )

print("\nDiagnostic complete. No files modified.")
