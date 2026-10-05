from pathlib import Path
from PIL import Image
import pandas as pd


# ==========================================
# Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_dataset"
)

ALIGNED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aligned_clinical_cxr.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cxr_manifest.csv"
)


# ==========================================
# Validate CXR images
# ==========================================

def preprocess_cxr():

    print("Starting CXR preprocessing...\n")

    # Load aligned multimodal data
    aligned = pd.read_csv(
        ALIGNED_FILE
    )

    aligned["subject_id"] = (
        aligned["subject_id"]
        .astype(str)
    )

    records = []

    total = len(aligned)

    valid = 0
    invalid = 0

    # --------------------------------------
    # Check every CXR
    # --------------------------------------

    for number, row in enumerate(
        aligned.itertuples(index=False),
        start=1
    ):

        subject_id = str(row.subject_id)
        cxr_file = str(row.cxr_file)

        image_path = (
            DATASET_DIR
            / subject_id
            / "cxr"
            / cxr_file
        )

        if not image_path.exists():

            invalid += 1
            continue

        try:

            # Open image
            with Image.open(image_path) as image:

                # Verify image
                image.verify()

            # Reopen after verify
            with Image.open(image_path) as image:

                width, height = image.size
                mode = image.mode

            records.append({
                "subject_id": subject_id,
                "dicom_id": row.dicom_id,
                "cxr_time": row.cxr_time,
                "cxr_file": cxr_file,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "image_mode": mode
            })

            valid += 1

        except Exception as e:

            invalid += 1

            print(
                f"Invalid image: "
                f"{image_path}"
            )

        # Progress
        if number % 1000 == 0:

            print(
                f"Checked {number:,}/{total:,} "
                f"images | "
                f"Valid: {valid:,} | "
                f"Invalid: {invalid:,}"
            )

    # --------------------------------------
    # Create manifest
    # --------------------------------------

    manifest = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------
    # Summary
    # --------------------------------------

    print("\n======================================")
    print("CXR preprocessing completed.")
    print("======================================")

    print(
        f"Total CXR records : {total:,}"
    )

    print(
        f"Valid images      : {valid:,}"
    )

    print(
        f"Invalid images    : {invalid:,}"
    )

    print(
        f"Output file       : {OUTPUT_FILE}"
    )

    if not manifest.empty:

        print("\nImage dimensions:")
        print(
            manifest[
                ["width", "height", "image_mode"]
            ]
            .value_counts()
            .to_string()
        )


if __name__ == "__main__":

    preprocess_cxr()