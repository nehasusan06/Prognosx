
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Prognosx")

SEQUENCE_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_manifest_labeled.csv"
)

# ============================================================
# RADIOGRAPHIC FINDINGS
# ============================================================

FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices"
]

# ============================================================
# LOAD LABELS
# ============================================================

print("\nLoading radiographic labels...")

labels_df = pd.read_csv(
    LABEL_FILE,
    usecols=["dicom_id"] + FINDINGS
)

# Verify that each DICOM ID has one label record.
if labels_df["dicom_id"].duplicated().any():
    raise ValueError(
        "Duplicate DICOM IDs found in the label file. "
        "Resolve duplicates before calculating class counts."
    )

# ============================================================
# LOAD TRAINING METADATA
# ============================================================

print("Loading TRAINING metadata...")

train_metadata = pd.read_csv(
    SEQUENCE_DIR / "train_metadata.csv"
)

if "target_dicom_id" not in train_metadata.columns:
    raise ValueError(
        "target_dicom_id is missing from train_metadata.csv"
    )

# ============================================================
# MATCH TRAINING DICOM IDS WITH LABELS
# ============================================================

merged = train_metadata[
    ["target_dicom_id"]
].merge(
    labels_df,
    left_on="target_dicom_id",
    right_on="dicom_id",
    how="left",
    validate="many_to_one"
)

print(f"\nTraining sequences: {len(merged)}")

missing = merged["dicom_id"].isna().sum()

print(f"Sequences without matched labels: {missing}")

# ============================================================
# TRAINING SET CLASS BALANCE
# ============================================================

print("\n" + "=" * 90)
print("TRAINING SET CLASS BALANCE")
print("=" * 90)

print(
    f"{'Finding':<30}"
    f"{'Positive':>12}"
    f"{'Negative':>12}"
    f"{'Unknown':>12}"
    f"{'Total':>12}"
    f"{'Positive %':>14}"
)

print("-" * 90)

for finding in FINDINGS:

    values = merged[finding]

    # Only explicit 0/1 labels are valid for this calculation.
    valid = values[values.isin([0.0, 1.0])]

    positive = int((valid == 1.0).sum())
    negative = int((valid == 0.0).sum())
    unknown = int(values.isna().sum())

    total = positive + negative

    positive_percent = (
        (positive / total) * 100 if total > 0 else 0
    )

    print(
        f"{finding:<30}"
        f"{positive:>12}"
        f"{negative:>12}"
        f"{unknown:>12}"
        f"{total:>12}"
        f"{positive_percent:>13.2f}%"
    )

print("=" * 90)
print("Class distribution check completed.")
