from pathlib import Path
import pandas as pd


# ==============================
# Project paths
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinical_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "selected_clinical_features.csv"
)


# ==============================
# MIMIC-IV feature mapping
# ==============================

FEATURE_MAP = {

    # Vital signs
    220045: "heart_rate",
    220179: "systolic_bp",
    220180: "diastolic_bp",
    220181: "mean_bp",
    220210: "respiratory_rate",
    220277: "spo2",

    # Temperature
    223761: "temperature_f",
    223762: "temperature_c",

    # Laboratory values
    50931: "glucose",
    50971: "potassium",
    50983: "sodium",
    51006: "bun",
    51222: "hemoglobin",
    51265: "platelets",
    51301: "wbc",
}


def process_clinical_data():

    print("Starting clinical feature selection...")
    print(f"Input: {INPUT_FILE}")

    selected_chunks = []

    total_rows = 0
    selected_rows = 0

    # Process large CSV in chunks
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=500_000,
            low_memory=False
        ),
        start=1
    ):

        total_rows += len(chunk)

        # Keep only selected item IDs
        chunk = chunk[
            chunk["itemid"].isin(FEATURE_MAP.keys())
        ].copy()

        if chunk.empty:
            print(
                f"Chunk {chunk_number}: "
                f"no selected features"
            )
            continue

        # Convert numerical value
        chunk["valuenum"] = pd.to_numeric(
            chunk["valuenum"],
            errors="coerce"
        )

        # Remove rows without usable numerical values
        chunk = chunk.dropna(
            subset=["valuenum"]
        )

        if chunk.empty:
            continue

        # Convert item ID to feature name
        chunk["feature"] = chunk["itemid"].map(
            FEATURE_MAP
        )

        # Keep required columns
        chunk = chunk[
            [
                "subject_id",
                "clinical_window",
                "event_type",
                "charttime",
                "itemid",
                "feature",
                "valuenum",
                "valueuom"
            ]
        ]

        selected_rows += len(chunk)

        selected_chunks.append(chunk)

        print(
            f"Chunk {chunk_number}: "
            f"processed {total_rows:,} rows | "
            f"selected {selected_rows:,}"
        )

    # ==============================
    # Combine selected data
    # ==============================

    if not selected_chunks:
        print("\nNo selected clinical features found.")
        return

    clinical_df = pd.concat(
        selected_chunks,
        ignore_index=True
    )

    # Convert timestamp
    clinical_df["charttime"] = pd.to_datetime(
        clinical_df["charttime"],
        errors="coerce"
    )

    # Sort chronologically
    clinical_df = clinical_df.sort_values(
        ["subject_id", "charttime"]
    )

    # Create output directory
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save
    clinical_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ==============================
    # Summary
    # ==============================

    print("\n======================================")
    print("Clinical feature selection completed.")
    print("======================================")

    print(
        f"Original records scanned : "
        f"{total_rows:,}"
    )

    print(
        f"Selected clinical records: "
        f"{len(clinical_df):,}"
    )

    print(
        f"Patients                 : "
        f"{clinical_df['subject_id'].nunique():,}"
    )

    print(
        f"Features                 : "
        f"{clinical_df['feature'].nunique()}"
    )

    print("\nFeature counts:")
    print(
        clinical_df["feature"]
        .value_counts()
        .to_string()
    )

    print(f"\nOutput file: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_clinical_data()