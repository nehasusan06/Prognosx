import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
FINAL_DATASET = PROJECT_ROOT / "data" / "processed" / "final_dataset"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "training_manifest.csv"

print("Loading training data...")

train = pd.read_csv(TRAIN_FILE)

print(f"Training rows: {len(train):,}")

# Find all patient order.csv files
order_files = list(FINAL_DATASET.glob("*/cxr/order.csv"))

print(f"Found {len(order_files):,} order files")

orders = []

for file in order_files:
    try:
        df = pd.read_csv(
            file,
            usecols=[
                "subject_id",
                "study_id",
                "dicom_id",
                "timestamp",
                "filename"
            ]
        )
        orders.append(df)
    except Exception:
        pass

if not orders:
    raise RuntimeError("No order.csv files were found.")

orders = pd.concat(orders, ignore_index=True)

# Convert IDs to strings
train["subject_id"] = train["subject_id"].astype(str)
train["dicom_id"] = train["dicom_id"].astype(str)

orders["subject_id"] = orders["subject_id"].astype(str)
orders["dicom_id"] = orders["dicom_id"].astype(str)

# Remove duplicate CXR mappings
orders = orders.drop_duplicates(
    subset=["subject_id", "dicom_id"]
)

# Merge study information into training data
manifest = train.merge(
    orders,
    on=["subject_id", "dicom_id"],
    how="left"
)

# Save manifest
manifest.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nTRAINING MANIFEST CREATED")
print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(manifest):,}")
print(f"Columns: {len(manifest.columns):,}")
print(f"Missing study_id: {manifest['study_id'].isna().sum():,}")

print("\nColumns:")
for column in manifest.columns:
    print(column)