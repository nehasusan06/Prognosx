import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Prognosx")

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_manifest_labeled.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_temporal_model.pt"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")

print("=" * 75)
print("RADIOGRAPHIC CLASSIFICATION EVALUATION")
print("=" * 75)

print("Device:", DEVICE)


# ============================================================
# EXACT TRAINED TRANSFORMER ARCHITECTURE
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
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

        x = self.output_layer(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained Transformer...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = TemporalTransformer()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()

print("Transformer loaded successfully.")


# ============================================================
# GENERATE FUTURE EMBEDDINGS
# ============================================================

def generate_predictions(split):

    print(
        f"\nGenerating {split} predictions..."
    )

    X = np.load(
        SEQUENCE_DIR
        / f"{split}_X.npy"
    )

    y = np.load(
        SEQUENCE_DIR
        / f"{split}_y.npy"
    )

    print(
        "Input shape:",
        X.shape
    )

    print(
        "Target embedding shape:",
        y.shape
    )

    # --------------------------------------------------------
    # Normalize clinical features exactly as during training
    # --------------------------------------------------------

    clinical_mean = checkpoint[
        "clinical_mean"
    ]

    clinical_std = checkpoint[
        "clinical_std"
    ]

    X = X.copy()

    X[:, :, 512:] = (
        X[:, :, 512:]
        - clinical_mean
    ) / (
        clinical_std + 1e-8
    )

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

            batch = batch.to(DEVICE)

            output = model(batch)

            predictions.append(
                output.cpu().numpy()
            )

    predictions = np.concatenate(
        predictions,
        axis=0
    )

    print(
        "Predictions:",
        predictions.shape
    )

    return predictions


# ============================================================
# LOAD LABEL MANIFEST
# ============================================================

print("\nLoading labeled manifest...")

labels_df = pd.read_csv(
    LABEL_FILE,
    usecols=[
        "dicom_id"
    ] + FINDINGS
)

labels_df["dicom_id"] = (
    labels_df["dicom_id"]
    .astype(str)
    .str.strip()
)

print(
    "Labeled CXR records:",
    len(labels_df)
)


# ============================================================
# GET LABELS FOR EACH SPLIT
# ============================================================

def get_labels(split):

    metadata = pd.read_csv(
        SEQUENCE_DIR
        / f"{split}_metadata.csv"
    )

    metadata["target_dicom_id"] = (
        metadata["target_dicom_id"]
        .astype(str)
        .str.strip()
    )

    merged = metadata[
        ["target_dicom_id"]
    ].merge(
        labels_df,
        left_on="target_dicom_id",
        right_on="dicom_id",
        how="left"
    )

    labeled_count = (
        merged["dicom_id"]
        .notna()
        .sum()
    )

    unlabeled_count = (
        merged["dicom_id"]
        .isna()
        .sum()
    )

    print(
        f"\n{split.upper()}:"
    )

    print(
        "  Total targets:",
        len(merged)
    )

    print(
        "  Labeled targets:",
        labeled_count
    )

    print(
        "  Unlabeled targets:",
        unlabeled_count
    )

    return merged[
        FINDINGS
    ].values.astype(
        np.float32
    )


# ============================================================
# GENERATE TRANSFORMER OUTPUTS
# ============================================================

print("\n" + "=" * 75)
print("GENERATING TRANSFORMER FUTURE EMBEDDINGS")
print("=" * 75)

train_pred = generate_predictions(
    "train"
)

val_pred = generate_predictions(
    "validation"
)

test_pred = generate_predictions(
    "test"
)


# ============================================================
# LOAD LABELS
# ============================================================

print("\n" + "=" * 75)
print("MATCHING RADIOGRAPHIC LABELS")
print("=" * 75)

train_labels = get_labels(
    "train"
)

val_labels = get_labels(
    "validation"
)

test_labels = get_labels(
    "test"
)


# ============================================================
# BINARY CLASSIFIER
# ============================================================

class BinaryClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer = nn.Linear(
            512,
            1
        )

    def forward(self, x):

        return self.layer(
            x
        ).squeeze(1)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(
    X_train,
    y_train
):

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True
    )

    classifier = BinaryClassifier()

    # --------------------------------------------------------
    # Handle class imbalance
    # --------------------------------------------------------

    positives = np.sum(
        y_train == 1
    )

    negatives = np.sum(
        y_train == 0
    )

    if positives > 0:

        pos_weight = (
            negatives
            / positives
        )

    else:

        pos_weight = 1.0

    loss_function = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight,
                dtype=torch.float32
            )
        )
    )

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=0.001
    )

    classifier.train()

    for epoch in range(15):

        for batch_X, batch_y in loader:

            optimizer.zero_grad()

            logits = classifier(
                batch_X
            )

            loss = loss_function(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

    return classifier


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

def predict_probabilities(
    classifier,
    X
):

    classifier.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=512,
        shuffle=False
    )

    probabilities = []

    with torch.no_grad():

        for (batch,) in loader:

            logits = classifier(
                batch
            )

            probabilities.append(
                torch.sigmoid(
                    logits
                ).numpy()
            )

    return np.concatenate(
        probabilities
    )


# ============================================================
# BASIC METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    tp = np.sum(
        (y_true == 1)
        &
        (predictions == 1)
    )

    tn = np.sum(
        (y_true == 0)
        &
        (predictions == 0)
    )

    fp = np.sum(
        (y_true == 0)
        &
        (predictions == 1)
    )

    fn = np.sum(
        (y_true == 1)
        &
        (predictions == 0)
    )

    total = len(y_true)

    accuracy = (
        (tp + tn) / total
        if total > 0
        else np.nan
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# AUROC
# ============================================================

def calculate_auroc(
    y_true,
    probabilities
):

    if len(
        np.unique(y_true)
    ) < 2:

        return np.nan

    order = np.argsort(
        -probabilities
    )

    y_sorted = (
        y_true[order]
    )

    positives = np.sum(
        y_sorted == 1
    )

    negatives = np.sum(
        y_sorted == 0
    )

    if (
        positives == 0
        or negatives == 0
    ):

        return np.nan

    tp = np.cumsum(
        y_sorted == 1
    )

    fp = np.cumsum(
        y_sorted == 0
    )

    tpr = tp / positives

    fpr = fp / negatives

    tpr = np.concatenate(
        [[0], tpr]
    )

    fpr = np.concatenate(
        [[0], fpr]
    )

    return np.trapezoid(
        tpr,
        fpr
    )


# ============================================================
# AUPRC
# ============================================================

def calculate_auprc(
    y_true,
    probabilities
):

    positives = np.sum(
        y_true == 1
    )

    if positives == 0:

        return np.nan

    order = np.argsort(
        -probabilities
    )

    y_sorted = (
        y_true[order]
    )

    tp = np.cumsum(
        y_sorted == 1
    )

    fp = np.cumsum(
        y_sorted == 0
    )

    precision = (
        tp
        / (tp + fp + 1e-12)
    )

    recall = (
        tp / positives
    )

    recall = np.concatenate(
        [[0], recall]
    )

    precision = np.concatenate(
        [[1], precision]
    )

    return np.trapezoid(
        precision,
        recall
    )


# ============================================================
# BEST THRESHOLD FROM VALIDATION
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(
        0.05,
        0.96,
        0.01
    ):

        (
            accuracy,
            precision,
            recall,
            f1
        ) = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    return best_threshold


# ============================================================
# EVALUATE ALL FINDINGS
# ============================================================

results = []

print("\n")
print("=" * 75)
print("RADIOGRAPHIC FINDING EVALUATION")
print("=" * 75)


for i, finding in enumerate(
    FINDINGS
):

    print("\n" + "-" * 75)

    print(
        f"[{i + 1}/14] {finding}"
    )

    print("-" * 75)

    # --------------------------------------------------------
    # Extract labels
    # --------------------------------------------------------

    y_train_all = (
        train_labels[:, i]
    )

    y_val_all = (
        val_labels[:, i]
    )

    y_test_all = (
        test_labels[:, i]
    )

    # --------------------------------------------------------
    # Only definite labels:
    #
    # 0 = negative
    # 1 = positive
    #
    # -1 = uncertain -> excluded
    # NaN = unavailable -> excluded
    # --------------------------------------------------------

    train_mask = np.isin(
        y_train_all,
        [0.0, 1.0]
    )

    val_mask = np.isin(
        y_val_all,
        [0.0, 1.0]
    )

    test_mask = np.isin(
        y_test_all,
        [0.0, 1.0]
    )

    X_train = train_pred[
        train_mask
    ]

    y_train = y_train_all[
        train_mask
    ]

    X_val = val_pred[
        val_mask
    ]

    y_val = y_val_all[
        val_mask
    ]

    X_test = test_pred[
        test_mask
    ]

    y_test = y_test_all[
        test_mask
    ]

    print(
        "Train labeled:",
        len(y_train)
    )

    print(
        "Validation labeled:",
        len(y_val)
    )

    print(
        "Test labeled:",
        len(y_test)
    )

    # --------------------------------------------------------
    # Check training classes
    # --------------------------------------------------------

    if len(y_train) == 0:

        print(
            "SKIPPED: no labeled training data."
        )

        continue

    if len(
        np.unique(y_train)
    ) < 2:

        print(
            "SKIPPED: only one class in training."
        )

        continue

    if len(y_val) == 0:

        print(
            "SKIPPED: no labeled validation data."
        )

        continue

    if len(y_test) == 0:

        print(
            "SKIPPED: no labeled test data."
        )

        continue

    # --------------------------------------------------------
    # Normalize embeddings
    # Training statistics only
    # --------------------------------------------------------

    mean = X_train.mean(
        axis=0
    )

    std = X_train.std(
        axis=0
    )

    std[
        std < 1e-8
    ] = 1.0

    X_train_scaled = (
        X_train - mean
    ) / std

    X_val_scaled = (
        X_val - mean
    ) / std

    X_test_scaled = (
        X_test - mean
    ) / std

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "Training classifier..."
    )

    classifier = train_classifier(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_probabilities = (
        predict_probabilities(
            classifier,
            X_val_scaled
        )
    )

    # --------------------------------------------------------
    # Threshold selected ONLY on validation
    # --------------------------------------------------------

    if len(
        np.unique(y_val)
    ) >= 2:

        threshold = (
            find_best_threshold(
                y_val,
                val_probabilities
            )
        )

    else:

        threshold = 0.50

        print(
            "Validation has one class; "
            "using threshold 0.50."
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_probabilities = (
        predict_probabilities(
            classifier,
            X_test_scaled
        )
    )

    (
        accuracy,
        precision,
        recall,
        f1
    ) = calculate_metrics(
        y_test,
        test_probabilities,
        threshold
    )

    auroc = calculate_auroc(
        y_test,
        test_probabilities
    )

    auprc = calculate_auprc(
        y_test,
        test_probabilities
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"AUROC     : {auroc:.4f}"
    )

    print(
        f"AUPRC     : {auprc:.4f}"
    )

    results.append({

        "Finding": finding,

        "Test_Samples": len(
            y_test
        ),

        "Positive_Cases": int(
            np.sum(y_test == 1)
        ),

        "Negative_Cases": int(
            np.sum(y_test == 0)
        ),

        "Threshold": threshold,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "AUROC": auroc,

        "AUPRC": auprc
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

if len(results_df) > 0:

    macro = {

        "Finding": "MACRO AVERAGE",

        "Test_Samples":
            results_df[
                "Test_Samples"
            ].sum(),

        "Positive_Cases":
            results_df[
                "Positive_Cases"
            ].sum(),

        "Negative_Cases":
            results_df[
                "Negative_Cases"
            ].sum(),

        "Threshold":
            results_df[
                "Threshold"
            ].mean(),

        "Accuracy":
            results_df[
                "Accuracy"
            ].mean(),

        "Precision":
            results_df[
                "Precision"
            ].mean(),

        "Recall":
            results_df[
                "Recall"
            ].mean(),

        "F1":
            results_df[
                "F1"
            ].mean(),

        "AUROC":
            results_df[
                "AUROC"
            ].mean(),

        "AUPRC":
            results_df[
                "AUPRC"
            ].mean()
    }

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame([macro])
        ],
        ignore_index=True
    )


# ============================================================
# SAVE
# ============================================================

output_file = (
    RESULTS_DIR
    / "radiographic_accuracy_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n\n")

print("=" * 110)
print("FINAL RADIOGRAPHIC CLASSIFICATION RESULTS")
print("=" * 110)

if len(results_df) > 0:

    print(
        results_df[
            [
                "Finding",
                "Test_Samples",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "AUROC",
                "AUPRC"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

else:

    print(
        "No valid findings were evaluated."
    )

print("=" * 110)

print(
    "\nResults saved to:"
)

print(
    output_file
)

print(
    "\nEVALUATION COMPLETE"
)