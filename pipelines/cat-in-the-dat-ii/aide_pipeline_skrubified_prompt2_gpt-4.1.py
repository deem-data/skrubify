import skrub
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Start DataOps plan
data = skrub.var("data", train_data).skb.subsample(n=100)

# Mark y and X
y = data["target"].skb.mark_as_y()
X = data.drop(["target", "id"], axis=1).skb.mark_as_X()

# List of columns by type
binary_cols = [col for col in train_data.columns if "bin" in col and col not in ["target", "id"]]
ordinal_cols = [col for col in train_data.columns if "ord" in col and col not in ["target", "id"]]
nominal_cols = [col for col in train_data.columns if "nom" in col and col not in ["target", "id"]]
cyclical_cols = ["day", "month"]

# Ordinal encoding for binary and ordinal features
ordinal_encoder = OrdinalEncoder()
X_ord = X.copy()
X_ord[binary_cols + ordinal_cols] = X_ord[binary_cols + ordinal_cols].skb.apply(ordinal_encoder)

# One-hot encoding for nominal features with low cardinality
low_cardinality_nom_cols = [col for col in nominal_cols if train_data[col].nunique() < 10]
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_low_card_nom = X_ord[low_cardinality_nom_cols].skb.apply(one_hot_encoder)
# X_low_card_nom is a numpy array, convert to DataFrame with correct index
X_low_card_nom_df = pd.DataFrame(X_low_card_nom, index=X_ord.index)

# Frequency encoding for nominal features with high cardinality
high_cardinality_nom_cols = [col for col in nominal_cols if train_data[col].nunique() >= 10]
def freq_encode(col, ref):
    freq = ref.value_counts(normalize=True)
    return col.map(freq)
for col in high_cardinality_nom_cols:
    X_ord[col] = X_ord[col].skb.apply_func(lambda s, ref=train_data[col]: freq_encode(s, ref))

# Combine all features
X_combined = pd.concat([X_ord, X_low_card_nom_df], axis=1).drop(low_cardinality_nom_cols, axis=1)

# Model
model = LGBMClassifier()
pred_proba = X_combined.skb.apply(model, y=y, predict_method="predict_proba")
# Select probability for class 1
pred = pred_proba.skb.apply_func(lambda arr: arr[:, 1])

# Split the data
splits = pred.skb.train_test_split(test_size=0.2, random_state=0)
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Predict on validation
valid_preds = learner.predict(splits["test"])
roc_auc = roc_auc_score(splits["y_test"], valid_preds)
print(f"Validation ROC AUC Score: {roc_auc}")

# Predict on test set
test_preds = learner.predict({"_skrub_X": test_data.drop("id", axis=1)})

# Save the predictions to a CSV file
output = pd.DataFrame({"id": test_data.id, "target": test_preds})
output.to_csv("./working/submission_skrub.csv", index=False)