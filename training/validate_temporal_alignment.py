
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "processed" / "sequences"
SRC = ROOT / "data" / "processed" / "multimodal"

EMBED_PATH = SRC / "test_cxr_embeddings.npy"
META_PATH = SRC / "test_metadata.csv"
X_PATH = SEQ / "test_X.npy"
Y_PATH = SEQ / "test_y.npy"
SEQ_META_PATH = SEQ / "test_metadata.csv"

PATIENT_ID = "10048001"
EMBED_DIM = 512


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


for path in [EMBED_PATH, META_PATH, X_PATH, Y_PATH, SEQ_META_PATH]:
    require_file(path)

source_embeddings = np.load(EMBED_PATH, mmap_mode="r")
source_meta = pd.read_csv(META_PATH)
X = np.load(X_PATH, mmap_mode="r")
y = np.load(Y_PATH, mmap_mode="r")
seq_meta = pd.read_csv(SEQ_META_PATH)

print("=" * 70)
print("PROGNOSX TEMPORAL SEQUENCE / DICOM ALIGNMENT VALIDATION")
print("=" * 70)
print("Source embeddings:", source_embeddings.shape)
print("Source metadata:", source_meta.shape)
print("Sequence X:", X.shape)
print("Sequence y:", y.shape)
print("Sequence metadata:", seq_meta.shape)

if source_embeddings.ndim != 2 or source_embeddings.shape[1] != EMBED_DIM:
    raise ValueError(f"Expected source embeddings [N,512], got {source_embeddings.shape}")

if len(source_embeddings) != len(source_meta):
    raise ValueError("Source embeddings and metadata row counts differ.")

if not (len(X) == len(y) == len(seq_meta)):
    raise ValueError("Sequence X, y and metadata row counts differ.")

required_source = {"subject_id", "dicom_id"}
required_sequence = {
    "subject_id", "target_dicom_id",
    "target_cxr_time", "sequence_start", "sequence_end"
}

if not required_source.issubset(source_meta.columns):
    raise ValueError(f"Missing source columns: {required_source - set(source_meta.columns)}")

if not required_sequence.issubset(seq_meta.columns):
    raise ValueError(f"Missing sequence columns: {required_sequence - set(seq_meta.columns)}")

time_col = next(
    (c for c in ["cxr_time", "target_cxr_time", "timestamp"]
     if c in source_meta.columns),
    None
)
if time_col is None:
    raise ValueError("No source timestamp column found.")

# Preserve each metadata row's original embedding-array index.
source_meta = source_meta.copy()
source_meta["_embedding_index"] = np.arange(len(source_meta))
source_meta["_time"] = pd.to_datetime(source_meta[time_col], errors="coerce")

seq_meta = seq_meta.copy()
seq_meta["_target_time"] = pd.to_datetime(seq_meta["target_cxr_time"], errors="coerce")
seq_meta["_start_time"] = pd.to_datetime(seq_meta["sequence_start"], errors="coerce")
seq_meta["_end_time"] = pd.to_datetime(seq_meta["sequence_end"], errors="coerce")

source_meta["_subject"] = source_meta["subject_id"].astype(str).str.strip()
source_meta["_dicom"] = source_meta["dicom_id"].astype(str).str.strip()
seq_meta["_subject"] = seq_meta["subject_id"].astype(str).str.strip()
seq_meta["_target_dicom"] = seq_meta["target_dicom_id"].astype(str).str.strip()

duplicate_times = source_meta.dropna(subset=["_time"]).duplicated(
    ["_subject", "_time"], keep=False
)
print("\nSource rows sharing a patient timestamp:", int(duplicate_times.sum()))

# Recreate the original sequence sort order.
ordered = source_meta.sort_values(
    ["subject_id", "_time"]
).reset_index(drop=True)

# Keep source embeddings in original order.
ordered_embeddings = np.asarray(source_embeddings)

expected = []
for subject, group in ordered.groupby("_subject", sort=False):
    indices = group.index.to_list()

    for i in range(max(0, len(indices) - 3)):
        inputs = indices[i:i + 3]
        target = indices[i + 3]

        expected.append({
            "subject": subject,
            "input_indices": inputs,
            "target_index": target,
            "input_ids": [ordered.loc[j, "_dicom"] for j in inputs],
            "target_id": ordered.loc[target, "_dicom"],
            "input_times": [ordered.loc[j, "_time"] for j in inputs],
            "target_time": ordered.loc[target, "_time"],
        })

expected_by_key = {}
for item in expected:
    key = (item["subject"], item["target_id"])
    expected_by_key.setdefault(key, []).append(item)

matched = 0
unmatched = []
ambiguous = []
input_mismatches = 0
target_mismatches = 0
temporal_violations = []
equal_time_rows = []
metadata_order_issues = []
details = []

for row in range(len(seq_meta)):
    m = seq_meta.iloc[row]
    key = (m["_subject"], m["_target_dicom"])
    candidates = expected_by_key.get(key, [])

    if not candidates:
        unmatched.append(row)
        continue

    if len(candidates) > 1:
        ambiguous.append(row)
        continue

    item = candidates[0]
    matched += 1

    if pd.notna(m["_target_time"]) and pd.notna(item["target_time"]):
        if m["_target_time"] != item["target_time"]:
            details.append(f"Row {row}: saved target timestamp differs from source.")

    # Verify the three input embeddings.
    for pos, sorted_idx in enumerate(item["input_indices"]):
        original_idx = int(ordered.loc[sorted_idx, "_embedding_index"])
        original = np.asarray(
            ordered_embeddings[original_idx], dtype=np.float32
        )
        saved = np.asarray(X[row, pos, :EMBED_DIM], dtype=np.float32)

        if not np.allclose(saved, original, rtol=1e-5, atol=1e-6):
            input_mismatches += 1
            details.append(
                f"Row {row}, input {pos + 1}: mismatch for "
                f"{item['input_ids'][pos]}, "
                f"max_abs_diff={np.max(np.abs(saved - original)):.8g}"
            )

    # FIX: copy read-only memmap slices before normalization.
    target_idx = int(ordered.loc[item["target_index"], "_embedding_index"])
    target = np.array(
        ordered_embeddings[target_idx], dtype=np.float32, copy=True
    )
    target = target / max(float(np.linalg.norm(target)), 1e-12)

    saved_target = np.array(y[row], dtype=np.float32, copy=True)
    saved_target = saved_target / max(
        float(np.linalg.norm(saved_target)), 1e-12
    )

    if not np.allclose(saved_target, target, rtol=1e-5, atol=1e-6):
        target_mismatches += 1
        details.append(
            f"Row {row}: target mismatch for {item['target_id']}, "
            f"max_abs_diff={np.max(np.abs(saved_target - target)):.8g}"
        )

    times = item["input_times"]
    target_time = item["target_time"]

    if all(pd.notna(t) for t in times) and pd.notna(target_time):
        if any(t > target_time for t in times):
            temporal_violations.append(row)
        if any(t == target_time for t in times):
            equal_time_rows.append(row)

    if (
        pd.notna(m["_target_time"])
        and pd.notna(m["_end_time"])
        and m["_target_time"] < m["_end_time"]
    ):
        metadata_order_issues.append(row)

print("\n" + "-" * 70)
print("VALIDATION SUMMARY")
print("-" * 70)
print("Sequence rows:", len(seq_meta))
print("Rows matched:", matched)
print("Rows with no reconstructed match:", len(unmatched))
print("Rows with ambiguous patient/target keys:", len(ambiguous))
print("Input embedding mismatches:", input_mismatches)
print("Target embedding mismatches:", target_mismatches)
print("Rows with input later than target:", len(temporal_violations))
print("Rows with equal input/target timestamps:", len(equal_time_rows))
print("Rows where target time precedes sequence_end:", len(metadata_order_issues))

if details:
    print("\nFirst alignment details:")
    for line in details[:30]:
        print("-", line)

if unmatched:
    print("\nFirst unmatched rows:")
    for row in unmatched[:20]:
        m = seq_meta.iloc[row]
        print(row, m["_subject"], m["_target_dicom"])

if ambiguous:
    print("\nAmbiguous rows:", ambiguous[:20])

if temporal_violations:
    print("\nTemporal violations:", temporal_violations[:20])

if equal_time_rows:
    print("\nEqual-timestamp rows:", equal_time_rows[:30])

if metadata_order_issues:
    print("\nMetadata ordering issues:", metadata_order_issues[:20])

# Patient-specific summary
patient_rows = np.flatnonzero(
    seq_meta["_subject"].to_numpy() == PATIENT_ID
)

print("\n" + "-" * 70)
print(f"PATIENT {PATIENT_ID} CHECK")
print("-" * 70)
print("Patient sequence count:", len(patient_rows))

for row in patient_rows:
    m = seq_meta.iloc[row]
    print(
        f"row={row} | target={m['_target_dicom']} | "
        f"start={m['_start_time']} | end={m['_end_time']} | "
        f"target_time={m['_target_time']}"
    )

print("\nValidation finished. No files were modified.")
print(
    "Equal timestamps remain ambiguous: this script cannot establish "
    "the true chronology of CXRs sharing the same timestamp."
)
