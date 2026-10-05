from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MULTIMODAL_DIR = DATA_DIR / "multimodal"
SEQUENCE_DIR = DATA_DIR / "sequences"

SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 3

# 11 clinical features
CLINICAL_FEATURES = [
    "bun",
    "diastolic_bp",
    "glucose",
    "heart_rate",
    "potassium",
    "respiratory_rate",
    "sodium",
    "spo2",
    "systolic_bp",
    "temperature_c",
    "temperature_f",
]


def create_sequences(split_name):

    print("=" * 60)
    print(f"CREATING {split_name.upper()} MULTIMODAL SEQUENCES")
    print("=" * 60)

    embedding_file = (
        MULTIMODAL_DIR
        / f"{split_name}_cxr_embeddings.npy"
    )

    metadata_file = (
        MULTIMODAL_DIR
        / f"{split_name}_metadata.csv"
    )

    # Load CXR embeddings
    embeddings = np.load(embedding_file)

    # Load clinical metadata
    metadata = pd.read_csv(metadata_file)

    print(f"CXR embeddings: {embeddings.shape}")
    print(f"Metadata rows: {len(metadata)}")

    # Check clinical columns
    missing_features = [
        feature
        for feature in CLINICAL_FEATURES
        if feature not in metadata.columns
    ]

    if missing_features:
        print("Missing clinical features:")
        print(missing_features)
        return

    # Convert time
    metadata["cxr_time"] = pd.to_datetime(
        metadata["cxr_time"],
        errors="coerce"
    )

    # Keep original embedding index
    metadata["embedding_index"] = np.arange(
        len(metadata)
    )

    # Sort chronologically for each patient
    metadata = metadata.sort_values(
        ["subject_id", "cxr_time"]
    ).reset_index(drop=True)

    sequence_inputs = []
    sequence_targets = []
    sequence_metadata = []

    # Process each patient separately
    for subject_id, patient_data in metadata.groupby(
        "subject_id"
    ):

        patient_data = patient_data.sort_values(
            "cxr_time"
        )

        indices = (
            patient_data["embedding_index"]
            .astype(int)
            .tolist()
        )

        # Need 3 previous visits + 1 future visit
        if len(indices) <= SEQUENCE_LENGTH:
            continue

        for i in range(
            len(indices) - SEQUENCE_LENGTH
        ):

            input_indices = indices[
                i:i + SEQUENCE_LENGTH
            ]

            target_index = indices[
                i + SEQUENCE_LENGTH
            ]

            # -----------------------------
            # CXR features
            # -----------------------------

            input_cxr = embeddings[
                input_indices
            ]

            # -----------------------------
            # Clinical features
            # -----------------------------

            input_clinical = (
                patient_data.iloc[
                    i:i + SEQUENCE_LENGTH
                ][CLINICAL_FEATURES]
                .to_numpy(dtype=np.float32)
            )

            # -----------------------------
            # Combine CXR + clinical
            # -----------------------------

            combined_sequence = np.concatenate(
                [
                    input_cxr,
                    input_clinical
                ],
                axis=1
            )

            # Expected:
            # 3 visits × 523 features

            sequence_inputs.append(
                combined_sequence.astype(
                    np.float32
                )
            )

            # Target = next CXR embedding
            target_embedding = embeddings[
                target_index
            ]

            sequence_targets.append(
                target_embedding.astype(
                    np.float32
                )
            )

            target_row = patient_data.iloc[
                i + SEQUENCE_LENGTH
            ]

            sequence_metadata.append({
                "subject_id": subject_id,
                "target_dicom_id": target_row[
                    "dicom_id"
                ],
                "target_cxr_time": target_row[
                    "cxr_time"
                ],
                "sequence_start": patient_data.iloc[
                    i
                ]["cxr_time"],
                "sequence_end": patient_data.iloc[
                    i + SEQUENCE_LENGTH - 1
                ]["cxr_time"]
            })

    if not sequence_inputs:

        print(
            "No valid temporal sequences found."
        )

        return

    X = np.asarray(
        sequence_inputs,
        dtype=np.float32
    )

    y = np.asarray(
        sequence_targets,
        dtype=np.float32
    )

    sequence_metadata = pd.DataFrame(
        sequence_metadata
    )

    # Save
    np.save(
        SEQUENCE_DIR
        / f"{split_name}_X.npy",
        X
    )

    np.save(
        SEQUENCE_DIR
        / f"{split_name}_y.npy",
        y
    )

    sequence_metadata.to_csv(
        SEQUENCE_DIR
        / f"{split_name}_metadata.csv",
        index=False
    )

    print()
    print("MULTIMODAL TEMPORAL SEQUENCES CREATED")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Sequences: {len(X)}")
    print(
        f"Features per visit: {X.shape[2]}"
    )
    print(
        f"Saved to: {SEQUENCE_DIR}"
    )


if __name__ == "__main__":

    create_sequences("train")

    create_sequences("validation")

    create_sequences("test")

    print("=" * 60)
    print("ALL MULTIMODAL TEMPORAL SEQUENCES CREATED")
    print("=" * 60)