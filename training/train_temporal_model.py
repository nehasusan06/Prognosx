import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

# Script is in C:\Prognosx\training
# Project root is C:\Prognosx
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "best_temporal_model_l2.pt"

SEQUENCE_LENGTH = 3
INPUT_DIM = 523
IMAGE_EMBEDDING_DIM = 512

CLINICAL_START = 512
CLINICAL_END = 523

D_MODEL = 256
N_HEADS = 8
NUM_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
SEED = 42


# ============================================================
# REPRODUCIBILITY AND DEVICE
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Sequence directory: {SEQUENCE_DIR}")


# ============================================================
# LOAD SEQUENCE ARRAYS
# ============================================================

def load_array(filename):
    path = SEQUENCE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find: {path}\n"
            "Check that your temporal sequence files exist."
        )

    array = np.load(path)
    print(f"Loaded {filename}: {array.shape}")

    return array


train_X = load_array("train_X.npy")
train_y = load_array("train_y.npy")

validation_X = load_array("validation_X.npy")
validation_y = load_array("validation_y.npy")

test_X = load_array("test_X.npy")
test_y = load_array("test_y.npy")


# ============================================================
# VALIDATE DATA SHAPES
# ============================================================

def validate_shapes(X, y, split_name):
    if X.ndim != 3:
        raise ValueError(
            f"{split_name}_X must be 3D. Got {X.shape}"
        )

    if X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected sequence length {SEQUENCE_LENGTH}, "
            f"got {X.shape[1]}"
        )

    if X.shape[2] != INPUT_DIM:
        raise ValueError(
            f"Expected {INPUT_DIM} input features, "
            f"got {X.shape[2]}"
        )

    if y.ndim != 2 or y.shape[1] != IMAGE_EMBEDDING_DIM:
        raise ValueError(
            f"{split_name}_y must have shape "
            f"[samples, {IMAGE_EMBEDDING_DIM}]. "
            f"Got {y.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{split_name}: X and y sample counts differ."
        )


validate_shapes(train_X, train_y, "train")
validate_shapes(validation_X, validation_y, "validation")
validate_shapes(test_X, test_y, "test")


# ============================================================
# CLINICAL FEATURE NORMALIZATION
# Fit normalization statistics on TRAINING DATA ONLY
# ============================================================

train_X = train_X.astype(np.float32)
validation_X = validation_X.astype(np.float32)
test_X = test_X.astype(np.float32)

train_clinical = train_X[
    :, :, CLINICAL_START:CLINICAL_END
]

clinical_mean = np.nanmean(
    train_clinical,
    axis=(0, 1)
)

clinical_mean = np.where(
    np.isfinite(clinical_mean),
    clinical_mean,
    0.0
)

train_clinical_filled = np.where(
    np.isfinite(train_clinical),
    train_clinical,
    clinical_mean.reshape(1, 1, -1)
)

clinical_std = np.std(
    train_clinical_filled,
    axis=(0, 1)
)

clinical_std = np.where(
    np.isfinite(clinical_std) & (clinical_std > 1e-6),
    clinical_std,
    1.0
)


def normalize_clinical_features(X):
    X = X.astype(np.float32, copy=True)

    clinical = X[
        :, :, CLINICAL_START:CLINICAL_END
    ]

    clinical = np.where(
        np.isfinite(clinical),
        clinical,
        clinical_mean.reshape(1, 1, -1)
    )

    clinical = (
        clinical - clinical_mean.reshape(1, 1, -1)
    ) / clinical_std.reshape(1, 1, -1)

    X[:, :, CLINICAL_START:CLINICAL_END] = clinical

    return X


train_X = normalize_clinical_features(train_X)

validation_X = normalize_clinical_features(
    validation_X
)

test_X = normalize_clinical_features(test_X)


# ============================================================
# NORMALIZE TARGET EMBEDDINGS
# BioMedCLIP embeddings are expected to be L2-normalized
# ============================================================

def normalize_targets(y):
    y = y.astype(np.float32, copy=True)

    norms = np.linalg.norm(
        y,
        axis=1,
        keepdims=True
    )

    return y / np.maximum(norms, 1e-12)


train_y = normalize_targets(train_y)
validation_y = normalize_targets(validation_y)
test_y = normalize_targets(test_y)


# ============================================================
# CREATE DATALOADERS
# ============================================================

def make_loader(X, y, shuffle=False):
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32))
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )


train_loader = make_loader(
    train_X, train_y, shuffle=True
)

validation_loader = make_loader(
    validation_X, validation_y
)

test_loader = make_loader(
    test_X, test_y
)


# ============================================================
# TEMPORAL TRANSFORMER MODEL
# Input: 3 visits, 523 features per visit
# Output: predicted 512-D embedding for the next visit
# ============================================================

class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        sequence_length=SEQUENCE_LENGTH,
        output_dim=IMAGE_EMBEDDING_DIM
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                sequence_length,
                d_model
            )
        )

        nn.init.normal_(
            self.positional_embedding,
            mean=0.0,
            std=0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Linear output layer without LayerNorm.
        self.output_layer = nn.Linear(
            d_model,
            output_dim
        )

    def forward(self, x):
        # x shape: [batch, 3, 523]

        x = self.input_projection(x)

        x = x + self.positional_embedding

        x = self.transformer_encoder(x)

        # Use the representation of the third input visit.
        x = x[:, -1, :]

        predicted_embedding = self.output_layer(x)

        # Match the unit-length BioMedCLIP embeddings.
        predicted_embedding = F.normalize(
            predicted_embedding,
            p=2,
            dim=-1,
            eps=1e-12
        )

        # Shape: [batch, 512]
        return predicted_embedding


model = TemporalTransformer().to(DEVICE)

print("\nTemporal Transformer initialized.")
print(model)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING AND VALIDATION
# ============================================================

def run_epoch(model, loader, optimizer=None):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        y_batch = y_batch.to(
            DEVICE,
            non_blocking=True
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            if training:
                loss.backward()
                optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


# ============================================================
# TRAINING LOOP WITH EARLY STOPPING
# ============================================================

best_validation_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
best_state = None

for epoch in range(1, EPOCHS + 1):

    train_loss = run_epoch(
        model,
        train_loader,
        optimizer=optimizer
    )

    validation_loss = run_epoch(
        model,
        validation_loader
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Validation MSE: {validation_loss:.6f}"
    )

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        best_state = copy.deepcopy(
            model.state_dict()
        )

        checkpoint = {
            "model_state_dict": best_state,
            "clinical_mean": clinical_mean,
            "clinical_std": clinical_std,
            "input_dim": INPUT_DIM,
            "output_dim": IMAGE_EMBEDDING_DIM,
            "sequence_length": SEQUENCE_LENGTH,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )

        print(
            f"Saved best checkpoint: {CHECKPOINT_PATH}"
        )

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            f"Early stopping at epoch {epoch}."
        )
        break


# ============================================================
# TEST EVALUATION
# ============================================================

if best_state is None:
    raise RuntimeError(
        "Training did not produce a valid checkpoint."
    )

model.load_state_dict(best_state)

test_loss = run_epoch(
    model,
    test_loader
)

print("\n========== TRAINING COMPLETE ==========")
print(f"Best epoch: {best_epoch}")
print(
    f"Best validation MSE: "
    f"{best_validation_loss:.6f}"
)
print(f"Test MSE: {test_loss:.6f}")
print(f"Checkpoint: {CHECKPOINT_PATH}")