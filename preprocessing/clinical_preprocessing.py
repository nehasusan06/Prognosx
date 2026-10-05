from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "final_dataset"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "clinical_features.csv"


def extract_clinical_data():

    all_data = []

    patient_dirs = [
        p for p in FINAL_DATASET_DIR.iterdir()
        if p.is_dir()
    ]

    total_patients = len(patient_dirs)

    print(f"Found {total_patients} patients.")
    print("Starting clinical preprocessing...\n")

    for patient_number, patient_dir in enumerate(patient_dirs, start=1):

        clinical_dir = patient_dir / "clinical"

        if not clinical_dir.exists():
            continue

        patient_data = []

        # Find all clinical CSV files
        for clinical_window in clinical_dir.iterdir():

            if not clinical_window.is_dir():
                continue

            # -------------------------
            # Chart events
            # -------------------------
            chartevents_file = clinical_window / "chartevents.csv"

            if chartevents_file.exists():

                try:
                    df = pd.read_csv(chartevents_file)

                    if not df.empty:

                        columns = [
                            "charttime",
                            "itemid",
                            "value",
                            "valuenum",
                            "valueuom"
                        ]

                        available = [
                            c for c in columns
                            if c in df.columns
                        ]

                        chart_df = df[available].copy()

                        chart_df["subject_id"] = patient_dir.name
                        chart_df["clinical_window"] = clinical_window.name
                        chart_df["event_type"] = "chart"

                        patient_data.append(chart_df)

                except Exception as e:
                    print(
                        f"Error reading {chartevents_file}: {e}"
                    )

            # -------------------------
            # Lab events
            # -------------------------
            labevents_file = clinical_window / "labevents.csv"

            if labevents_file.exists():

                try:
                    df = pd.read_csv(labevents_file)

                    if not df.empty:

                        columns = [
                            "charttime",
                            "itemid",
                            "value",
                            "valuenum",
                            "valueuom"
                        ]

                        available = [
                            c for c in columns
                            if c in df.columns
                        ]

                        lab_df = df[available].copy()

                        lab_df["subject_id"] = patient_dir.name
                        lab_df["clinical_window"] = clinical_window.name
                        lab_df["event_type"] = "lab"

                        patient_data.append(lab_df)

                except Exception as e:
                    print(
                        f"Error reading {labevents_file}: {e}"
                    )

        # Combine this patient's data
        if patient_data:
            patient_combined = pd.concat(
                patient_data,
                ignore_index=True
            )

            all_data.append(patient_combined)

        # Progress
        if patient_number % 10 == 0 or patient_number == total_patients:
            print(
                f"Processed {patient_number}/{total_patients} patients"
            )

    # -------------------------
    # Combine all patients
    # -------------------------

    if not all_data:
        print("\nNo clinical data found.")
        return

    clinical_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # Convert timestamps
    if "charttime" in clinical_df.columns:
        clinical_df["charttime"] = pd.to_datetime(
            clinical_df["charttime"],
            errors="coerce"
        )

    # Convert numeric values
    if "valuenum" in clinical_df.columns:
        clinical_df["valuenum"] = pd.to_numeric(
            clinical_df["valuenum"],
            errors="coerce"
        )

    # Remove rows without item IDs
    clinical_df = clinical_df.dropna(
        subset=["itemid"]
    )

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save
    clinical_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===================================")
    print("Clinical preprocessing completed.")
    print("===================================")
    print(f"Total records : {len(clinical_df):,}")
    print(
        f"Patients      : "
        f"{clinical_df['subject_id'].nunique():,}"
    )
    print(f"Output file   : {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_clinical_data()