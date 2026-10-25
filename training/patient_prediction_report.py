
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "data" / "processed" / "sequences"
CHECKPOINT_PATH = ROOT / "models" / "best_temporal_model_l2.pt"
METADATA_PATH = SEQ_DIR / "test_metadata.csv"

PATIENT_ID = "10048001"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLINICAL_START = 512
CLINICAL_END = 523


# ============================================================
# MODEL: MUST MATCH train_temporal_model.py
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

        self.input_projection = nn.Linear(input_dim, d_model)

        self.positional_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

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

        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.positional_embedding
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.output_layer(x)
        return F.normalize(x, p=2, dim=-1, eps=1e-12)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

clinical_mean = np.asarray(checkpoint["clinical_mean"], dtype=np.float32)
clinical_std = np.asarray(checkpoint["clinical_std"], dtype=np.float32)

if clinical_mean.shape != (11,) or clinical_std.shape != (11,):
    raise ValueError("Checkpoint must contain 11 clinical means and std values.")

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

# Strict loading: do not silently ignore mismatched model parameters.
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

print(f"Checkpoint loaded: {CHECKPOINT_PATH}")
print(f"Best epoch: {checkpoint.get('best_epoch', 'unknown')}")


# ============================================================
# LOAD TEST DATA AND MATCH METADATA
# ============================================================

X = np.load(SEQ_DIR / "test_X.npy").astype(np.float32)
y = np.load(SEQ_DIR / "test_y.npy").astype(np.float32)
metadata = pd.read_csv(METADATA_PATH)

if not (len(X) == len(y) == len(metadata)):
    raise ValueError(
        f"Row counts differ: X={len(X)}, y={len(y)}, "
        f"metadata={len(metadata)}. Refusing unsafe alignment."
    )

required = {"subject_id", "target_dicom_id", "target_cxr_time"}
missing = required - set(metadata.columns)
if missing:
    raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

# Compare patient IDs as strings to avoid integer/string mismatches.
patient_mask = metadata["subject_id"].astype(str).str.strip() == PATIENT_ID
rows = np.flatnonzero(patient_mask)

if len(rows) == 0:
    raise ValueError(
        f"Patient {PATIENT_ID} is not present in test_metadata.csv."
    )

# Use saved row positions to keep X, y and metadata aligned.
clinical = X[:, :, CLINICAL_START:CLINICAL_END].copy()
clinical = np.where(
    np.isfinite(clinical),
    clinical,
    clinical_mean.reshape(1, 1, -1),
)
clinical = (
    clinical - clinical_mean.reshape(1, 1, -1)
) / clinical_std.reshape(1, 1, -1)

X[:, :, CLINICAL_START:CLINICAL_END] = clinical

if not np.isfinite(X[rows]).all():
    raise ValueError("Non-finite values remain in this patient's input.")

# The training script normalized targets before calculating loss.
target_norms = np.linalg.norm(y, axis=1, keepdims=True)
y = y / np.maximum(target_norms, 1e-12)


# ============================================================
# PREDICT AND REPORT
# ============================================================

report_rows = []
embedding_rows = []

with torch.no_grad():
    for idx in rows:
        input_tensor = torch.from_numpy(X[idx:idx + 1]).to(DEVICE)
        predicted = model(input_tensor).cpu().numpy()[0]
        actual = y[idx]

        mse = float(np.mean((predicted - actual) ** 2))
        cosine = float(
            np.dot(predicted, actual)
            / max(np.linalg.norm(predicted) * np.linalg.norm(actual), 1e-12)
        )

        meta = metadata.iloc[idx]
        report_rows.append({
            "test_row": int(idx),
            "subject_id": str(meta["subject_id"]),
            "target_dicom_id": str(meta["target_dicom_id"]),
            "target_cxr_time": str(meta["target_cxr_time"]),
            "sequence_start": str(meta.get("sequence_start", "")),
            "sequence_end": str(meta.get("sequence_end", "")),
            "embedding_mse": mse,
            "cosine_similarity": cosine,
        })

        for dim, (pred_value, actual_value) in enumerate(zip(predicted, actual)):
            embedding_rows.append({
                "test_row": int(idx),
                "target_dicom_id": str(meta["target_dicom_id"]),
                "embedding_dimension": dim,
                "predicted_value": float(pred_value),
                "actual_value": float(actual_value),
                "absolute_error": float(abs(pred_value - actual_value)),
            })

report = pd.DataFrame(report_rows)
embedding_report = pd.DataFrame(embedding_rows)

output_dir = ROOT / "reports" / f"patient_{PATIENT_ID}"
output_dir.mkdir(parents=True, exist_ok=True)

report_path = output_dir / "patient_prediction_summary.csv"
embedding_path = output_dir / "embedding_comparison.csv"

report.to_csv(report_path, index=False)
embedding_report.to_csv(embedding_path, index=False)

print("\n" + "=" * 65)
print(f"PATIENT EMBEDDING PREDICTION REPORT: {PATIENT_ID}")
print("=" * 65)
print(f"Matching test sequences: {len(report)}")
print(f"Mean embedding MSE: {report['embedding_mse'].mean():.6f}")
print(f"Mean cosine similarity: {report['cosine_similarity'].mean():.6f}")
print("\nPer-target results:")
print(report.to_string(index=False))
print("\nFiles saved:")
print(report_path)
print(embedding_path)
print("\nNote: these metrics evaluate embedding prediction, not disease diagnosis.")
