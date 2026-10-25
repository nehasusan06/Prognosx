from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MULTIMODAL_DIR = (
    PROJECT_ROOT / "data" / "processed" / "multimodal"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "best_temporal_model_l2.pt"
)

SEQUENCE_LENGTH = 3
INPUT_DIM = 523
EMBEDDING_DIM = 512

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

# Choose the split to demonstrate prediction.
# "test" evaluates a held-out patient sequence.
SPLIT_NAME = "test"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# MODEL — SAME ARCHITECTURE AS TRAINING
# ============================================================

class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=523,
        d_model=256,
        n_heads=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        sequence_length=3,
        output_dim=512,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model,
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )

        nn.init.normal_(
            self.positional_embedding,
            mean=0.0,
            std=0.02,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(
            d_model,
            output_dim,
        )

    def forward(self, x):

        x = self.input_projection(x)
        x = x + self.positional_embedding
        x = self.transformer_encoder(x)
        x = x[:, -1, :]

        predicted = self.output_layer(x)

        return F.normalize(
            predicted,
            p=2,
            dim=-1,
            eps=1e-12,
        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

clinical_mean = np.asarray(
    checkpoint["clinical_mean"],
    dtype=np.float32,
)

clinical_std = np.asarray(
    checkpoint["clinical_std"],
    dtype=np.float32,
)

clinical_std = np.where(
    np.isfinite(clinical_std) & (clinical_std > 1e-6),
    clinical_std,
    1.0,
)

model = TemporalTransformer(
    input_dim=checkpoint["input_dim"],
    d_model=checkpoint["d_model"],
    n_heads=checkpoint["n_heads"],
    num_layers=checkpoint["num_layers"],
    dim_feedforward=checkpoint["dim_feedforward"],
    dropout=checkpoint["dropout"],
    sequence_length=checkpoint["sequence_length"],
    output_dim=checkpoint["output_dim"],
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint loaded:", CHECKPOINT_PATH)


# ============================================================
# LOAD EMBEDDINGS AND METADATA
# ============================================================

embedding_path = (
    MULTIMODAL_DIR
    / f"{SPLIT_NAME}_cxr_embeddings.npy"
)

metadata_path = (
    MULTIMODAL_DIR
    / f"{SPLIT_NAME}_metadata.csv"
)

embeddings = np.load(embedding_path).astype(np.float32)
metadata = pd.read_csv(metadata_path)

if len(embeddings) != len(metadata):
    raise ValueError(
        "Embedding and metadata row counts differ. "
        "Cannot safely match visits."
    )

required_columns = [
    "subject_id",
    "dicom_id",
    "cxr_time",
] + CLINICAL_FEATURES

missing = [
    column for column in required_columns
    if column not in metadata.columns
]

if missing:
    raise ValueError(
        f"Missing metadata columns: {missing}"
    )

if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
    raise ValueError(
        f"Expected embeddings with shape [N, 512], "
        f"got {embeddings.shape}"
    )

metadata["cxr_time"] = pd.to_datetime(
    metadata["cxr_time"],
    errors="coerce",
)

metadata["original_embedding_index"] = np.arange(
    len(metadata)
)

metadata = metadata.dropna(
    subset=["subject_id", "cxr_time"]
)

print("Embeddings:", embeddings.shape)
print("Metadata rows:", len(metadata))


# ============================================================
# FIND A PATIENT WITH AT LEAST FOUR VISITS
# ============================================================

metadata = metadata.sort_values(
    ["subject_id", "cxr_time"]
)

patient_groups = []

for subject_id, group in metadata.groupby("subject_id"):

    group = group.sort_values("cxr_time")

    if len(group) >= 4:
        patient_groups.append((subject_id, group))

if not patient_groups:
    raise ValueError(
        "No patient with at least four visits was found "
        f"in the {SPLIT_NAME} split."
    )

# Select the first eligible patient for demonstration.
subject_id, patient_data = patient_groups[0]

first_four = patient_data.iloc[:4].copy()

print("\nSelected patient:", subject_id)
print("\nFirst four chronological visits:")

print(
    first_four[
        ["dicom_id", "cxr_time"]
    ].to_string(index=False)
)


# ============================================================
# BUILD THE FIRST THREE VISITS
# ============================================================

input_indices = first_four[
    "original_embedding_index"
].iloc[:3].to_numpy(dtype=int)

target_index = int(
    first_four["original_embedding_index"].iloc[3]
)

input_cxr = embeddings[input_indices]

input_clinical = first_four[
    CLINICAL_FEATURES
].iloc[:3].to_numpy(dtype=np.float32)

# Replace missing/non-finite clinical measurements
# with the corresponding training mean.
input_clinical = np.where(
    np.isfinite(input_clinical),
    input_clinical,
    clinical_mean.reshape(1, -1),
)

# Apply the training normalization statistics.
input_clinical = (
    input_clinical - clinical_mean.reshape(1, -1)
) / clinical_std.reshape(1, -1)

# Combine 512 image features + 11 clinical features.
input_sequence = np.concatenate(
    [input_cxr, input_clinical],
    axis=1,
).astype(np.float32)

if input_sequence.shape != (3, 523):
    raise ValueError(
        f"Unexpected input shape: {input_sequence.shape}"
    )

if not np.isfinite(input_sequence).all():
    raise ValueError(
        "Input sequence contains non-finite values."
    )


# ============================================================
# PREDICT FOURTH VISIT EMBEDDING
# ============================================================

input_tensor = torch.from_numpy(
    input_sequence
).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    predicted_embedding = model(
        input_tensor
    ).cpu().numpy()[0]

# Actual fourth-visit embedding is used only for evaluation.
actual_embedding = embeddings[target_index].copy()

# Normalize the target to match the model output.
actual_norm = np.linalg.norm(actual_embedding)

if actual_norm > 1e-12:
    actual_embedding = actual_embedding / actual_norm

# Metrics
mse = np.mean(
    (predicted_embedding - actual_embedding) ** 2
)

cosine_similarity = np.dot(
    predicted_embedding,
    actual_embedding,
) / max(
    np.linalg.norm(predicted_embedding)
    * np.linalg.norm(actual_embedding),
    1e-12,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FOURTH VISIT EMBEDDING PREDICTION")
print("=" * 60)

print("Patient:", subject_id)

print("\nInput visits:")
for _, row in first_four.iloc[:3].iterrows():
    print(f"  {row['cxr_time']} | {row['dicom_id']}")

print("\nTarget fourth visit:")
print(f"  {first_four.iloc[3]['cxr_time']}")
print(f"  {first_four.iloc[3]['dicom_id']}")

print("\nPrediction:")
print("Predicted embedding shape:", predicted_embedding.shape)
print("First 10 predicted values:")
print(predicted_embedding[:10])

print("\nEvaluation against actual fourth-visit embedding:")
print(f"MSE: {mse:.6f}")
print(f"Cosine similarity: {cosine_similarity:.6f}")

print(
    "\nPredicted embedding norm:",
    f"{np.linalg.norm(predicted_embedding):.6f}",
)

print(
    "Actual embedding norm:",
    f"{np.linalg.norm(actual_embedding):.6f}",
)

print("\nPrediction completed.")