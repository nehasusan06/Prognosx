import glob
import numpy as np
import pandas as pd

for f in sorted(glob.glob("data/**/*.*", recursive=True)):
    try:
        if f.endswith(".npy"):
            a = np.load(f, allow_pickle=True)
            print(f, a.shape, a.dtype)
        elif f.endswith(".npz"):
            z = np.load(f, allow_pickle=True)
            print(f, {k: z[k].shape for k in z.files})
        elif f.endswith(".csv"):
            print(f, pd.read_csv(f, nrows=2).columns.tolist())
    except Exception as e:
        print(f, "ERR", e)