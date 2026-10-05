from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset
FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "final_dataset"

# Dataset manifest
TRAINING_MANIFEST = FINAL_DATASET_DIR / "training_manifest.csv"

# Project directories
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "training"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
EXPLAINABILITY_DIR = PROJECT_ROOT / "explainability"
APP_DIR = PROJECT_ROOT / "app"