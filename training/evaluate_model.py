from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_temporal_model.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("TRANSFORMER MODEL EVALUATION")
print("=" * 60)

print(f"Device: {DEVICE}")


# ============================================================
# MODEL
# ============================================================

class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=523,
        d_model=256,
        n_heads=8,
        num_layers=3,
        ff_dim=512,
        dropout=0.1
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                3,
                d_model
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu"
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )

        self.output_layer = nn.Sequential(
            nn.Linear(
                d_model,
                512
            ),
            nn.LayerNorm(512)
        )


    def forward(self, x):

        x = self.input_projection(x)

        x = (
            x
            + self.position_embedding
        )

        x = self.transformer(x)

        x = x[:, -1, :]

        return self.output_layer(x)


# ============================================================
# LOAD TEST DATA
# ============================================================

X = np.load(
    SEQUENCE_DIR / "test_X.npy"
)

y = np.load(
    SEQUENCE_DIR / "test_y.npy"
)


# ============================================================
# NORMALIZE CLINICAL FEATURES
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

clinical_mean = checkpoint[
    "clinical_mean"
]

clinical_std = checkpoint[
    "clinical_std"
]

X = X.copy()

X[:, :, 512:523] = (
    X[:, :, 512:523]
    - clinical_mean
) / clinical_std


# ============================================================
# TORCH DATA
# ============================================================

X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.float32
)

dataset = TensorDataset(
    X,
    y
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# LOAD MODEL
# ============================================================

model = TemporalTransformer()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

predictions = []
targets = []

with torch.no_grad():

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(DEVICE)

        output = model(
            X_batch
        )

        predictions.append(
            output.cpu().numpy()
        )

        targets.append(
            y_batch.numpy()
        )


predictions = np.concatenate(
    predictions,
    axis=0
)

targets = np.concatenate(
    targets,
    axis=0
)


# ============================================================
# MSE
# ============================================================

mse = np.mean(
    (predictions - targets) ** 2
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

numerator = np.sum(
    predictions * targets,
    axis=1
)

denominator = (
    np.linalg.norm(
        predictions,
        axis=1
    )
    *
    np.linalg.norm(
        targets,
        axis=1
    )
)

denominator = np.maximum(
    denominator,
    1e-8
)

cosine_similarity = (
    numerator / denominator
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test MSE: {mse:.6f}"
)

print(
    f"Mean Cosine Similarity: "
    f"{cosine_similarity.mean():.6f}"
)

print(
    f"Median Cosine Similarity: "
    f"{np.median(cosine_similarity):.6f}"
)

print(
    f"Minimum Cosine Similarity: "
    f"{cosine_similarity.min():.6f}"
)

print(
    f"Maximum Cosine Similarity: "
    f"{cosine_similarity.max():.6f}"
)

print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)