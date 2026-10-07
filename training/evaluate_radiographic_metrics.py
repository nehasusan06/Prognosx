import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Prognosx")

SEQUENCE_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"

LABEL_FILE = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "training_manifest_labeled.csv"
)

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "best_temporal_model.pt"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# RADIOGRAPHIC FINDINGS
# ============================================================

FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices"
]


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
        dim_feedforward=512,
        dropout=0.1,
        output_dim=512
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(
            d_model,
            output_dim
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.transformer(x)

        x = x[:, -1, :]

        x = self.output_layer(x)

        return x


# ============================================================
# LOAD TRAINED TRANSFORMER
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED TRANSFORMER")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

model = TemporalTransformer()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# FUNCTION: GENERATE TRANSFORMER PREDICTIONS
# ============================================================

def generate_predictions(split):

    print(f"\nGenerating predictions for {split}...")

    X = np.load(
        SEQUENCE_DIR / f"{split}_X.npy"
    )

    y = np.load(
        SEQUENCE_DIR / f"{split}_y.npy"
    )

    # Clinical features are the final 11 columns
    clinical_mean = checkpoint["clinical_mean"]
    clinical_std = checkpoint["clinical_std"]

    X = X.copy()

    X[:, :, 512:] = (
        X[:, :, 512:] - clinical_mean
    ) / (clinical_std + 1e-8)

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False
    )

    predictions = []

    with torch.no_grad():

        for (batch,) in loader:

            output = model(batch)

            predictions.append(
                output.numpy()
            )

    predictions = np.concatenate(
        predictions,
        axis=0
    )

    print(
        f"{split}: {len(predictions)} predictions generated"
    )

    return predictions, y


# ============================================================
# LOAD LABELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING RADIOGRAPHIC LABELS")
print("=" * 70)

label_columns = ["dicom_id"] + FINDINGS

labels_df = pd.read_csv(
    LABEL_FILE,
    usecols=label_columns
)

print(
    f"Label records: {len(labels_df)}"
)

duplicate_ids = labels_df["dicom_id"].duplicated().sum()

print(
    f"Duplicate DICOM IDs: {duplicate_ids}"
)

if duplicate_ids > 0:
    raise ValueError(
        "Duplicate dicom_id values found in "
        "training_manifest_labeled.csv"
    )


# ============================================================
# MATCH TARGET DICOM IDs WITH LABELS
# ============================================================

def get_labels(split):

    metadata = pd.read_csv(
        SEQUENCE_DIR /
        f"{split}_metadata.csv"
    )

    print(
        f"{split}: {len(metadata)} sequences"
    )

    merged = metadata[
        ["target_dicom_id"]
    ].merge(
        labels_df,
        left_on="target_dicom_id",
        right_on="dicom_id",
        how="left",
        validate="one_to_one"
    )

    missing = merged["dicom_id"].isna().sum()

    print(
        f"{split}: missing label records = {missing}"
    )

    if missing > 0:
        raise ValueError(
            f"{missing} target DICOM IDs could not "
            f"be matched with labels."
        )

    return merged[FINDINGS].values.astype(
        np.float32
    )


# ============================================================
# LOAD ALL SPLITS
# ============================================================

train_pred, train_actual = generate_predictions(
    "train"
)

val_pred, val_actual = generate_predictions(
    "validation"
)

test_pred, test_actual = generate_predictions(
    "test"
)

train_labels = get_labels("train")
val_labels = get_labels("validation")
test_labels = get_labels("test")


# ============================================================
# CHECK SHAPES
# ============================================================

print("\n" + "=" * 70)
print("DATA CHECK")
print("=" * 70)

print("Train predictions :", train_pred.shape)
print("Validation predictions :", val_pred.shape)
print("Test predictions :", test_pred.shape)

print("Train labels :", train_labels.shape)
print("Validation labels :", val_labels.shape)
print("Test labels :", test_labels.shape)


# ============================================================
# FIND BEST THRESHOLD USING VALIDATION SET
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return best_threshold


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    # AUROC requires both classes
    if len(np.unique(y_true)) == 2:

        auroc = roc_auc_score(
            y_true,
            probabilities
        )

        auprc = average_precision_score(
            y_true,
            probabilities
        )

    else:

        auroc = np.nan
        auprc = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc
    }


# ============================================================
# TRAIN CLASSIFIERS AND EVALUATE
# ============================================================

results = []

print("\n" + "=" * 70)
print("RADIOGRAPHIC CLASSIFICATION")
print("=" * 70)


for i, finding in enumerate(FINDINGS):

    print(
        f"\n[{i + 1}/14] {finding}"
    )

    # --------------------------------------------------------
    # TRAINING LABELS
    # --------------------------------------------------------

    train_y = train_labels[:, i]

    # Keep ONLY 0 and 1
    train_mask = np.isin(
        train_y,
        [0.0, 1.0]
    )

    X_train = train_pred[
        train_mask
    ]

    y_train = train_y[
        train_mask
    ].astype(int)

    # --------------------------------------------------------
    # VALIDATION LABELS
    # --------------------------------------------------------

    val_y = val_labels[:, i]

    val_mask = np.isin(
        val_y,
        [0.0, 1.0]
    )

    X_val = val_pred[
        val_mask
    ]

    y_val = val_y[
        val_mask
    ].astype(int)

    # --------------------------------------------------------
    # TEST LABELS
    # --------------------------------------------------------

    test_y = test_labels[:, i]

    test_mask = np.isin(
        test_y,
        [0.0, 1.0]
    )

    X_test = test_pred[
        test_mask
    ]

    y_test = test_y[
        test_mask
    ].astype(int)

    print(
        f"Train samples: {len(y_train)}"
    )

    print(
        f"Validation samples: {len(y_val)}"
    )

    print(
        f"Test samples: {len(y_test)}"
    )

    # Need both classes for training
    if len(np.unique(y_train)) < 2:

        print(
            "Skipped - only one class in training data."
        )

        continue

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # LOGISTIC REGRESSION CLASSIFIER
    # --------------------------------------------------------

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    classifier.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # VALIDATION PROBABILITIES
    # --------------------------------------------------------

    val_probabilities = classifier.predict_proba(
        X_val_scaled
    )[:, 1]

    # Select threshold ONLY using validation
    threshold = find_best_threshold(
        y_val,
        val_probabilities
    )

    # --------------------------------------------------------
    # TEST PROBABILITIES
    # --------------------------------------------------------

    test_probabilities = classifier.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # FINAL TEST METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold
    )

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1        : {metrics['f1']:.4f}"
    )

    print(
        f"AUROC     : {metrics['auroc']:.4f}"
    )

    print(
        f"AUPRC     : {metrics['auprc']:.4f}"
    )

    results.append({

        "finding": finding,

        "test_samples": len(y_test),

        "positive_cases": int(
            np.sum(y_test == 1)
        ),

        "threshold": threshold,

        "accuracy": metrics["accuracy"],

        "precision": metrics["precision"],

        "recall": metrics["recall"],

        "f1": metrics["f1"],

        "auroc": metrics["auroc"],

        "auprc": metrics["auprc"]
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# MACRO AVERAGE
# ============================================================

macro_results = {

    "finding": "MACRO AVERAGE",

    "test_samples": results_df[
        "test_samples"
    ].sum(),

    "positive_cases": results_df[
        "positive_cases"
    ].sum(),

    "threshold": results_df[
        "threshold"
    ].mean(),

    "accuracy": results_df[
        "accuracy"
    ].mean(),

    "precision": results_df[
        "precision"
    ].mean(),

    "recall": results_df[
        "recall"
    ].mean(),

    "f1": results_df[
        "f1"
    ].mean(),

    "auroc": results_df[
        "auroc"
    ].mean(),

    "auprc": results_df[
        "auprc"
    ].mean()
}


results_df = pd.concat(
    [
        results_df,
        pd.DataFrame([macro_results])
    ],
    ignore_index=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    RESULTS_DIR /
    "radiographic_metrics.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n\n" + "=" * 100)
print("FINAL RADIOGRAPHIC CLASSIFICATION RESULTS")
print("=" * 100)

display_columns = [
    "finding",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc"
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 100)

print(
    f"Results saved to:\n{output_file}"
)

print("=" * 100)
