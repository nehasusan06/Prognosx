from pathlib import Path
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

CLINICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "selected_clinical_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aligned_clinical_cxr.csv"
)


# ==========================================
# Main alignment function
# ==========================================

def create_temporal_alignment():

    print("Loading selected clinical features...")

    clinical = pd.read_csv(
        CLINICAL_FILE
    )

    clinical["subject_id"] = (
        clinical["subject_id"]
        .astype(str)
    )

    clinical["charttime"] = pd.to_datetime(
        clinical["charttime"],
        errors="coerce"
    )

    print(
        f"Clinical records loaded: "
        f"{len(clinical):,}"
    )

    # --------------------------------------
    # Aggregate multiple measurements
    # belonging to the same CXR window
    # --------------------------------------

    clinical_agg = (
        clinical
        .groupby(
            [
                "subject_id",
                "clinical_window",
                "feature"
            ],
            as_index=False
        )
        .agg(
            value=("valuenum", "mean")
        )
    )

    print(
        f"Aggregated clinical records: "
        f"{len(clinical_agg):,}"
    )

    # --------------------------------------
    # Convert long format → wide format
    # --------------------------------------

    clinical_wide = (
        clinical_agg
        .pivot_table(
            index=[
                "subject_id",
                "clinical_window"
            ],
            columns="feature",
            values="value",
            aggfunc="mean"
        )
        .reset_index()
    )

    clinical_wide.columns.name = None

    print(
        f"Clinical feature vectors: "
        f"{len(clinical_wide):,}"
    )

    # --------------------------------------
    # Read CXR order files
    # --------------------------------------

    cxr_records = []

    patient_dirs = [
        p for p in DATASET_DIR.iterdir()
        if p.is_dir()
    ]

    print(
        f"Found {len(patient_dirs)} patient folders."
    )

    for number, patient_dir in enumerate(
        patient_dirs,
        start=1
    ):

        order_file = (
            patient_dir
            / "cxr"
            / "order.csv"
        )

        if not order_file.exists():
            continue

        try:

            order = pd.read_csv(
                order_file
            )

            if order.empty:
                continue

            # Normalize column names
            order.columns = [
                str(c).strip()
                for c in order.columns
            ]

            # Expected columns based on
            # the verified order.csv structure
            if len(order.columns) >= 7:

                for _, row in order.iterrows():

                    cxr_records.append({
                        "subject_id": str(
                            row.iloc[1]
                        ),
                        "dicom_id": str(
                            row.iloc[4]
                        ),
                        "cxr_time": row.iloc[5],
                        "cxr_file": str(
                            row.iloc[7]
                        )
                    })

        except Exception as e:

            print(
                f"Error reading "
                f"{order_file}: {e}"
            )

        if number % 500 == 0:

            print(
                f"Scanned {number}/"
                f"{len(patient_dirs)} "
                f"patients"
            )

    # --------------------------------------
    # Create CXR dataframe
    # --------------------------------------

    cxr = pd.DataFrame(
        cxr_records
    )

    if cxr.empty:

        print("No CXR records found.")
        return

    cxr["cxr_time"] = pd.to_datetime(
        cxr["cxr_time"],
        errors="coerce"
    )

    print(
        f"\nCXR records found: "
        f"{len(cxr):,}"
    )

    # --------------------------------------
    # Match CXR DICOM ID with
    # clinical_window
    # --------------------------------------

    aligned = cxr.merge(
        clinical_wide,
        left_on=[
            "subject_id",
            "dicom_id"
        ],
        right_on=[
            "subject_id",
            "clinical_window"
        ],
        how="left"
    )

    # Remove duplicate clinical_window column
    if "clinical_window" in aligned.columns:

        aligned = aligned.drop(
            columns=["clinical_window"]
        )

    # --------------------------------------
    # Save
    # --------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    aligned.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------
    # Summary
    # --------------------------------------

    print("\n======================================")
    print("Temporal alignment completed.")
    print("======================================")

    print(
        f"CXR records              : "
        f"{len(aligned):,}"
    )

    print(
        f"Patients                  : "
        f"{aligned['subject_id'].nunique():,}"
    )

    print(
        f"CXR records with clinical "
        f"data                     : "
        f"{aligned['dicom_id'].notna().sum():,}"
    )

    print(
        f"Output file              : "
        f"{OUTPUT_FILE}"
    )

    print("\nColumns:")
    print(
        aligned.columns.tolist()
    )


if __name__ == "__main__":

    create_temporal_alignment()