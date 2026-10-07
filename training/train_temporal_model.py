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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32

# Keep this at 1 for the first test run.
# We will increase it after confirming everything works.
EPOCHS = 20

LEARNING_RATE = 1e-4

D_MODEL = 256
N_HEADS = 8
NUM_LAYERS = 3
FF_DIM = 512

DROPOUT = 0.1

PATIENCE = 5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("TEMPORAL MODEL TRAINING")
print("=" * 60)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
else:
    print("Running on CPU")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_X = np.load(
    SEQUENCE_DIR / "train_X.npy"
)

train_y = np.load(
    SEQUENCE_DIR / "train_y.npy"
)

val_X = np.load(
    SEQUENCE_DIR / "validation_X.npy"
)

val_y = np.load(
    SEQUENCE_DIR / "validation_y.npy"
)

test_X = np.load(
    SEQUENCE_DIR / "test_X.npy"
)

test_y = np.load(
    SEQUENCE_DIR / "test_y.npy"
)


print("\nDataset shapes:")

print("Train X:", train_X.shape)
print("Train y:", train_y.shape)

print("Validation X:", val_X.shape)
print("Validation y:", val_y.shape)

print("Test X:", test_X.shape)
print("Test y:", test_y.shape)


# ============================================================
# NORMALIZE CLINICAL FEATURES
# ============================================================
#
# Features 0:512   = CXR embeddings
# Features 512:523 = clinical features
#
# Statistics are calculated ONLY from training data.
# The same statistics are then applied to validation/test.
# ============================================================

print("\nNormalizing clinical features...")

clinical_train = train_X[:, :, 512:523]

clinical_mean = clinical_train.mean(
    axis=(0, 1)
)

clinical_std = clinical_train.std(
    axis=(0, 1)
)

# Avoid division by zero
clinical_std[
    clinical_std < 1e-6
] = 1.0


def normalize_clinical(X):

    X = X.copy()

    X[:, :, 512:523] = (
        X[:, :, 512:523]
        - clinical_mean
    ) / clinical_std

    return X


train_X = normalize_clinical(train_X)
val_X = normalize_clinical(val_X)
test_X = normalize_clinical(test_X)


# ============================================================
# CONVERT TO TORCH
# ============================================================

train_X = torch.tensor(
    train_X,
    dtype=torch.float32
)

train_y = torch.tensor(
    train_y,
    dtype=torch.float32
)

val_X = torch.tensor(
    val_X,
    dtype=torch.float32
)

val_y = torch.tensor(
    val_y,
    dtype=torch.float32
)

test_X = torch.tensor(
    test_X,
    dtype=torch.float32
)

test_y = torch.tensor(
    test_y,
    dtype=torch.float32
)


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = TensorDataset(
    train_X,
    train_y
)

val_dataset = TensorDataset(
    val_X,
    val_y
)

test_dataset = TensorDataset(
    test_X,
    test_y
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# TEMPORAL TRANSFORMER MODEL
# ============================================================

class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=523,
        d_model=256,
        n_heads=8,
        num_layers=3,
        ff_dim=512,
        dropout=0.1,
        output_dim=512
    ):

        super().__init__()

        # Project 523 features to model dimension
        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        # Learnable positional information
        # Sequence length = 3
        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                3,
                d_model
            )
        )

        # Transformer encoder
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

        # Predict future CXR embedding
        self.output_layer = nn.Sequential(
            nn.Linear(
                d_model,
                512
            ),
            nn.LayerNorm(512)
        )


    def forward(self, x):

        # Input:
        # [batch, 3, 523]

        x = self.input_projection(x)

        x = (
            x
            + self.position_embedding
        )

        x = self.transformer(x)

        # Use final temporal state
        x = x[:, -1, :]

        output = self.output_layer(x)

        return output


# ============================================================
# CREATE MODEL
# ============================================================

model = TemporalTransformer(
    input_dim=523,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
    output_dim=512
)

model = model.to(DEVICE)


print("\nModel:")
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

epochs_without_improvement = 0

best_model_path = (
    MODEL_DIR
    / "best_temporal_model.pt"
)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * X_batch.size(0)
        )


    train_loss /= len(
        train_loader.dataset
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            val_loss += (
                loss.item()
                * X_batch.size(0)
            )


    val_loss /= len(
        val_loader.dataset
    )


    print(
        f"Epoch "
        f"{epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "clinical_mean":
                    clinical_mean,

                "clinical_std":
                    clinical_std,

                "input_dim":
                    523,

                "output_dim":
                    512,

                "sequence_length":
                    3,

                "best_val_loss":
                    best_val_loss
            },
            best_model_path
        )

        print(
            "  ✓ Best model saved"
        )

    else:

        epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("TESTING BEST MODEL")
print("=" * 60)


# PyTorch 2.6+ changed the default of weights_only.
# Our checkpoint contains NumPy arrays as well as model weights.
# This checkpoint was created by our own trusted script.
checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

test_loss = 0.0

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch
        )

        test_loss += (
            loss.item()
            * X_batch.size(0)
        )


test_loss /= len(
    test_loader.dataset
)


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    f"Test MSE: {test_loss:.6f}"
)

print(
    f"Best validation MSE: "
    f"{best_val_loss:.6f}"
)

print(
    f"\nModel saved to:"
    f"\n{best_model_path}"
)

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)