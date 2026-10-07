import numpy as np

train_embeddings = np.load(r"C:\Prognosx\data\processed\multimodal\train_cxr_embeddings.npy")

print("Embedding shape:", train_embeddings.shape)

# Check if embeddings are L2-normalized (unit vectors)
norms = np.linalg.norm(train_embeddings, axis=1)
print(f"Embedding norm — mean: {norms.mean():.4f}, std: {norms.std():.4f}")

# Random (unrelated) pair cosine similarity
rng = np.random.default_rng(42)
n_samples = 5000
idx_a = rng.integers(0, len(train_embeddings), n_samples)
idx_b = rng.integers(0, len(train_embeddings), n_samples)
a = train_embeddings[idx_a]
b = train_embeddings[idx_b]
cos_sim = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

print(f"\nRandom (unrelated) pair cosine similarity:")
print(f"  Mean:   {cos_sim.mean():.4f}")
print(f"  Median: {np.median(cos_sim):.4f}")
print(f"  Min:    {cos_sim.min():.4f}")
print(f"  Max:    {cos_sim.max():.4f}")
print(f"  Std:    {cos_sim.std():.4f}")

print(f"\nFor comparison:")
print(f"  Your baseline (same-patient, last-observed) cosine similarity: 0.8800")
print(f"  Your trained model cosine similarity:                          0.9204")