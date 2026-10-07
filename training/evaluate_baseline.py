from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)


def evaluate_baseline(split_name):

    print("=" * 60)
    print(f"{split_name.upper()} BASELINE")
    print("=" * 60)

    X = np.load(
        SEQUENCE_DIR / f"{split_name}_X.npy"
    )

    y = np.load(
        SEQUENCE_DIR / f"{split_name}_y.npy"
    )

    # Last observed CXR embedding
    # Each visit contains:
    # 512 CXR features + 11 clinical features
    #
    # Therefore CXR features are [:, :, :512]
    last_cxr = X[:, -1, :512]

    # MSE
    mse = np.mean(
        (last_cxr - y) ** 2
    )

    # Cosine similarity
    numerator = np.sum(
        last_cxr * y,
        axis=1
    )

    denominator = (
        np.linalg.norm(last_cxr, axis=1)
        * np.linalg.norm(y, axis=1)
    )

    denominator = np.maximum(
        denominator,
        1e-8
    )

    cosine_similarity = (
        numerator / denominator
    )

    print(f"Sequences: {len(X)}")
    print(f"Baseline MSE: {mse:.6f}")
    print(
        f"Mean Cosine Similarity: "
        f"{cosine_similarity.mean():.6f}"
    )


if __name__ == "__main__":

    evaluate_baseline("train")
    evaluate_baseline("validation")
    evaluate_baseline("test")

    print("=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)