from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDING_DIR = DATA_DIR / "embeddings"
OUTPUT_DIR = DATA_DIR / "multimodal"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_FILE = EMBEDDING_DIR / "cxr_embeddings.npy"
EMBEDDING_METADATA_FILE = (
    EMBEDDING_DIR / "cxr_embedding_metadata.csv"
)

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


def build_split(split_name, clinical_file):

    print("=" * 60)
    print(f"BUILDING {split_name.upper()} MULTIMODAL DATASET")
    print("=" * 60)

    # Load clinical data
    print("Loading clinical data...")

    clinical = pd.read_csv(clinical_file)

    print(f"Clinical rows: {len(clinical)}")

    # Load all BioMedCLIP embeddings
    print("Loading CXR embeddings...")

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # Load embedding metadata
    print("Loading embedding metadata...")

    metadata = pd.read_csv(
        EMBEDDING_METADATA_FILE
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    # Create matching key
    clinical["match_key"] = (
        clinical["subject_id"].astype(str)
        + "_"
        + clinical["dicom_id"].astype(str)
    )

    metadata["match_key"] = (
        metadata["subject_id"].astype(str)
        + "_"
        + metadata["dicom_id"].astype(str)
    )

    # Map each CXR to its embedding index
    embedding_index = pd.Series(
        np.arange(len(metadata)),
        index=metadata["match_key"]
    )

    clinical["embedding_index"] = (
        clinical["match_key"].map(
            embedding_index
        )
    )

    before = len(clinical)

    clinical = clinical.dropna(
        subset=["embedding_index"]
    ).copy()

    after = len(clinical)

    print(
        f"Matched embeddings: {after}/{before}"
    )

    # Get embedding rows
    indices = (
        clinical["embedding_index"]
        .astype(int)
        .to_numpy()
    )

    selected_embeddings = embeddings[
        indices
    ]

    # Remove temporary columns
    clinical = clinical.drop(
        columns=[
            "match_key",
            "embedding_index"
        ]
    )

    # Save embeddings
    output_embeddings = (
        OUTPUT_DIR
        / f"{split_name}_cxr_embeddings.npy"
    )

    np.save(
        output_embeddings,
        selected_embeddings.astype(
            np.float32
        )
    )

    # Save clinical metadata
    output_metadata = (
        OUTPUT_DIR
        / f"{split_name}_metadata.csv"
    )

    clinical.to_csv(
        output_metadata,
        index=False
    )

    print(
        f"Saved embeddings: {output_embeddings}"
    )

    print(
        f"Saved metadata: {output_metadata}"
    )

    print(
        f"Final embedding shape: "
        f"{selected_embeddings.shape}"
    )


if __name__ == "__main__":

    build_split(
        "train",
        TRAIN_FILE
    )

    build_split(
        "validation",
        VAL_FILE
    )

    build_split(
        "test",
        TEST_FILE
    )

    print("=" * 60)
    print("MULTIMODAL DATASET CREATION COMPLETE")
    print("=" * 60)