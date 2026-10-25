from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

# Script location: C:\Prognosx\training
# Project root:    C:\Prognosx

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "sequences"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "best_temporal_model_l2.pt"
)

BATCH_SIZE = 64

CLINICAL_START = 512
CLINICAL_END = 523

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Sequence directory: {SEQUENCE_DIR}")


# ============================================================
# TEMPORAL TRANSFORMER
# Must match the training architecture
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
            torch.zeros(
                1,
                sequence_length,
                d_model,
            )
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

        # Predict the 512-dimensional image embedding.
        self.output_layer = nn.Linear(
            d_model,
            output_dim,
        )

    def forward(self, x):

        # Input: [batch, 3, 523]
        x = self.input_projection(x)

        x = x + self.positional_embedding

        x = self.transformer_encoder(x)

        # Select the encoded representation of the third visit.
        x = x[:, -1, :]

        predicted_embedding = self.output_layer(x)

        # L2-normalize the predicted embedding.
        predicted_embedding = F.normalize(
            predicted_embedding,
            p=2,
            dim=-1,
            eps=1e-12,
        )

        # Output: [batch, 512]
        return predicted_embedding


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}\n"
        "Run train_temporal_model.py first."
    )

# Load only checkpoints that you trust, since this checkpoint
# contains NumPy arrays in addition to model parameters.
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

# Validate checkpoint normalization dimensions.
if clinical_mean.shape != (11,):
    raise ValueError(
        f"Expected 11 clinical means, got {clinical_mean.shape}"
    )

if clinical_std.shape != (11,):
    raise ValueError(
        f"Expected 11 clinical standard deviations, "
        f"got {clinical_std.shape}"
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

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nCheckpoint loaded successfully.")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(
    f"Best training epoch: "
    f"{checkpoint.get('best_epoch', 'unknown')}"
)
print(
    f"Best validation MSE: "
    f"{checkpoint.get('best_validation_loss', float('nan')):.6f}"
)


# ============================================================
# LOAD TEST ARRAYS
# ============================================================

def load_array(filename):

    path = SEQUENCE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find sequence file:\n{path}"
        )

    array = np.load(path)

    print(f"Loaded {filename}: {array.shape}")

    return array


test_X = load_array("test_X.npy").astype(np.float32)
test_y = load_array("test_y.npy").astype(np.float32)


# ============================================================
# VALIDATE TEST SHAPES
# ============================================================

expected_sequence_length = checkpoint["sequence_length"]
expected_input_dim = checkpoint["input_dim"]
expected_output_dim = checkpoint["output_dim"]

if test_X.ndim != 3:
    raise ValueError(
        f"test_X must be 3D. Received {test_X.shape}"
    )

if test_y.ndim != 2:
    raise ValueError(
        f"test_y must be 2D. Received {test_y.shape}"
    )

if test_X.shape[1] != expected_sequence_length:
    raise ValueError(
        f"Sequence length mismatch: {test_X.shape[1]} "
        f"vs {expected_sequence_length}"
    )

if test_X.shape[2] != expected_input_dim:
    raise ValueError(
        f"Input dimension mismatch: {test_X.shape[2]} "
        f"vs {expected_input_dim}"
    )

if test_y.shape[1] != expected_output_dim:
    raise ValueError(
        f"Target dimension mismatch: {test_y.shape[1]} "
        f"vs {expected_output_dim}"
    )

if len(test_X) != len(test_y):
    raise ValueError(
        "The number of test inputs and targets does not match."
    )


# ============================================================
# HANDLE NON-FINITE INPUT VALUES
# ============================================================

if not np.isfinite(test_X[:, :, :CLINICAL_START]).all():
    raise ValueError(
        "Non-finite values found in image embeddings. "
        "Check the saved sequence arrays."
    )

if not np.isfinite(test_y).all():
    raise ValueError(
        "Non-finite values found in target embeddings. "
        "Check the saved sequence arrays."
    )


# ============================================================
# APPLY TRAINING CLINICAL NORMALIZATION
# Use the mean and standard deviation saved in the checkpoint.
# Do not calculate new statistics from the test set.
# ============================================================

clinical = test_X[
    :, :, CLINICAL_START:CLINICAL_END
].copy()

clinical = np.where(
    np.isfinite(clinical),
    clinical,
    clinical_mean.reshape(1, 1, -1),
)

clinical = (
    clinical - clinical_mean.reshape(1, 1, -1)
) / clinical_std.reshape(1, 1, -1)

test_X[:, :, CLINICAL_START:CLINICAL_END] = clinical

if not np.isfinite(test_X).all():
    raise ValueError(
        "Non-finite values remain in test inputs after normalization."
    )


# ============================================================
# NORMALIZE TARGET EMBEDDINGS
# ============================================================

target_norms_before = np.linalg.norm(
    test_y,
    axis=1,
    keepdims=True,
)

test_y = test_y / np.maximum(
    target_norms_before,
    1e-12,
)


# ============================================================
# CREATE TEST DATALOADER
# ============================================================

test_dataset = TensorDataset(
    torch.from_numpy(test_X),
    torch.from_numpy(test_y),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# RUN EVALUATION
# ============================================================

all_predictions = []
all_targets = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        predictions = model(X_batch)

        all_predictions.append(
            predictions.cpu().numpy()
        )

        all_targets.append(
            y_batch.cpu().numpy()
        )


predictions = np.concatenate(
    all_predictions,
    axis=0,
)

targets = np.concatenate(
    all_targets,
    axis=0,
)


# ============================================================
# CALCULATE METRICS
# ============================================================

# Mean squared error across all test samples and dimensions.
mse = np.mean(
    (predictions - targets) ** 2
)

# Embedding norms.
prediction_norms = np.linalg.norm(
    predictions,
    axis=1,
)

target_norms = np.linalg.norm(
    targets,
    axis=1,
)

# Cosine similarity for each test sample.
dot_products = np.sum(
    predictions * targets,
    axis=1,
)

denominators = np.maximum(
    prediction_norms * target_norms,
    1e-12,
)

cosine_similarity = (
    dot_products / denominators
)

# Mean and median squared error per sample.
sample_mse = np.mean(
    (predictions - targets) ** 2,
    axis=1,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print("TEMPORAL EMBEDDING PREDICTION RESULTS")

print(
    "============================================================"
)

print(f"Test samples: {len(targets)}")

print(f"Test MSE: {mse:.6f}")

print(
    f"Mean sample MSE: {sample_mse.mean():.6f}"
)

print(
    f"Median sample MSE: {np.median(sample_mse):.6f}"
)

print(
    f"Mean cosine similarity: "
    f"{cosine_similarity.mean():.6f}"
)

print(
    f"Median cosine similarity: "
    f"{np.median(cosine_similarity):.6f}"
)

print(
    f"Minimum cosine similarity: "
    f"{cosine_similarity.min():.6f}"
)

print(
    f"Maximum cosine similarity: "
    f"{cosine_similarity.max():.6f}"
)

print(
    f"Mean predicted embedding norm: "
    f"{prediction_norms.mean():.6f}"
)

print(
    f"Mean target embedding norm: "
    f"{target_norms.mean():.6f}"
)

print(
    "============================================================"
)

print("Evaluation completed successfully.")