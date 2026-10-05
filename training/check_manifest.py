import pandas as pd

file_path = "training_manifest_labeled.csv"

df = pd.read_csv(file_path)

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
for col in df.columns:
    print(col)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DATA TYPES =====")
print(df.dtypes)