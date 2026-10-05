from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
import open_clip

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "cxr_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_FILE = OUTPUT_DIR / "cxr_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "cxr_embedding_metadata.csv"


def extract_features():

    print("=" * 60)
    print("PROGNO SX - BIOMEDCLIP FEATURE EXTRACTION")
    print("=" * 60)

    df = pd.read_csv(MANIFEST_FILE)

    print(f"Total images: {len(df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading BioMedCLIP...")

    model, preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    model = model.to(device)
    model.eval()

    print("BioMedCLIP loaded.")

    # Check for previous checkpoint
    checkpoint_file = OUTPUT_DIR / "checkpoint_embeddings.npy"
    checkpoint_metadata = OUTPUT_DIR / "checkpoint_metadata.csv"

    if checkpoint_file.exists() and checkpoint_metadata.exists():

        embeddings = list(
            np.load(checkpoint_file)
        )

        metadata = pd.read_csv(
            checkpoint_metadata
        )

        start_index = len(metadata)

        print()
        print("RESUMING PREVIOUS EXTRACTION")
        print(f"Already completed: {start_index}")
        print(f"Remaining: {len(df) - start_index}")

    else:

        embeddings = []
        metadata = pd.DataFrame()
        start_index = 0

        print()
        print("Starting extraction from image 1.")

    with torch.no_grad():

        for i in range(start_index, len(df)):

            row = df.iloc[i]

            image_path = Path(row["image_path"])

            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                image_tensor = (
                    preprocess(image)
                    .unsqueeze(0)
                    .to(device)
                )

                feature = model.encode_image(
                    image_tensor
                )

                feature = feature / feature.norm(
                    dim=-1,
                    keepdim=True
                )

                embeddings.append(
                    feature.cpu().numpy()[0]
                )

                metadata = pd.concat(
                    [
                        metadata,
                        pd.DataFrame([row])
                    ],
                    ignore_index=True
                )

                print(
                    f"Processed {i + 1}/{len(df)}"
                )

                # SAVE CHECKPOINT EVERY 500 IMAGES
                if (i + 1) % 500 == 0:

                    np.save(
                        checkpoint_file,
                        np.asarray(
                            embeddings,
                            dtype=np.float32
                        )
                    )

                    metadata.to_csv(
                        checkpoint_metadata,
                        index=False
                    )

                    print(
                        f"CHECKPOINT SAVED: {i + 1}"
                    )

            except Exception as e:

                print(
                    f"Skipping image {i}: {image_path}"
                )

                print(
                    f"Reason: {e}"
                )

    # Final save
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    metadata.to_csv(
        METADATA_FILE,
        index=False
    )

    print("=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    extract_features()