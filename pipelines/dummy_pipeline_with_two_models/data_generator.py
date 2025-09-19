import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# --- Generate training data ---
n_train = 500
train_df = pd.DataFrame({
    "id": np.arange(1, n_train + 1),
    "feat1": np.random.randn(n_train) * 10,
    "feat2": np.random.randn(n_train) * 5,
    "label1": np.random.randint(0, 2, n_train),  # binary target
    "label2": np.random.randint(0, 2, n_train)   # binary target
})

# --- Generate test data ---
n_test = 200
test_df = pd.DataFrame({
    "id": np.arange(1001, 1001 + n_test),
    "feat1": np.random.randn(n_test) * 10,
    "feat2": np.random.randn(n_test) * 5,
})

# --- Save to CSVs ---
os.makedirs("./input", exist_ok=True)
os.makedirs("./working", exist_ok=True)
train_df.to_csv("./input/train.csv", index=False)
test_df.to_csv("./input/test.csv", index=False)

print("Dummy train.csv and X_test.csv created in ./input/")
