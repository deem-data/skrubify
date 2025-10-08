import pandas as pd
import numpy as np
import skrub
from skrub import selectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


# Skrub requires custom transformers to be scikit-learn compatible.
# This FrequencyEncoder mimics the logic from the original pipeline.
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_encoder_ = None

    def fit(self, X: pd.DataFrame, y=None):
        # Skrub applies this transformer to one column at a time.
        # The input X is a DataFrame with a single column.
        col = X.columns[0]
        self.freq_encoder_ = X[col].value_counts(normalize=True)
        return self

    def transform(self, X: pd.DataFrame):
        col = X.columns[0]
        # .map will introduce NaNs for unseen values, which matches the original behavior.
        return pd.DataFrame({col: X[col].map(self.freq_encoder_)})


# --- Skrub DataOps Pipeline Definition ---

# Load data and start the DataOps plan with a skrub variable.
# Subsampling is used for faster previews and is ignored during final execution.
train_data = skrub.var("train_data", pd.read_csv("./input/train.csv")).skb.subsample(
    n=100
)

# Separate target and features, marking them for the learner.
y = train_data["target"].skb.mark_as_y()
X = train_data.drop(["target", "id"], axis=1).skb.mark_as_X()

# Define selectors to identify column groups instead of manually listing them.
binary_ordinal_selector = selectors.filter_names(
    lambda name: "bin" in name or "ord" in name
)
low_card_nom_selector = selectors.filter(
    lambda col: "nom" in col.name and col.nunique() < 10
)
high_card_nom_selector = selectors.filter(
    lambda col: "nom" in col.name and col.nunique() >= 10
)
# Selector for columns that are not transformed (e.g., cyclical).
other_cols_selector = selectors.filter_names(
    lambda name: "bin" not in name and "ord" not in name and "nom" not in name
)

# Add operations to the plan for each column group.
# Ordinal encoding for binary and ordinal features.
bin_ord_cols = X.skb.select(binary_ordinal_selector)
bin_ord_cols_encoded = bin_ord_cols.skb.apply(OrdinalEncoder())

# One-hot encoding for low-cardinality nominal features.
low_card_nom_cols = X.skb.select(low_card_nom_selector)
low_card_nom_cols_encoded = low_card_nom_cols.skb.apply(
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)

# Frequency encoding for high-cardinality nominal features using the custom transformer.
high_card_nom_cols = X.skb.select(high_card_nom_selector)
high_card_nom_cols_encoded = high_card_nom_cols.skb.apply(FrequencyEncoder())

# Select the remaining columns that are kept as is.
other_cols = X.skb.select(other_cols_selector)

# Combine all processed feature groups into a single table.
X_processed = other_cols.skb.concat(
    [bin_ord_cols_encoded, high_card_nom_cols_encoded, low_card_nom_cols_encoded],
    axis=1,
)

# Define the model and prediction steps in the plan.
model = LGBMClassifier(random_state=42)
# Specify predict_method to get probabilities.
pred_proba = X_processed.skb.apply(model, y=y, predict_method="predict_proba")
# Slice the output to get the probability of the positive class (class 1).
pred = pred_proba[:, 1]

# --- Pipeline Execution ---

# The pipeline is now fully defined. Let's execute it.
# Get train/validation splits from the final step of the plan.
splits = pred.skb.train_test_split(train_size=0.8, test_size=0.2, random_state=0)

# Create a trainable learner object from the plan.
learner = pred.skb.make_learner()

# Train the learner. Skrub executes the entire defined pipeline on the training data.
learner.fit(splits["train"])

# Predict on the validation set and evaluate.
valid_preds = learner.predict(splits["test"])
roc_auc = roc_auc_score(splits["y_test"], valid_preds)
print(f"Validation ROC AUC Score: {roc_auc}")

# Predict on the test set.
test_data = pd.read_csv("./input/test.csv")
# Pass the raw test features. The learner automatically applies all preprocessing steps.
test_preds = learner.predict({"_skrub_X": test_data.drop("id", axis=1)})

# Save the predictions to a CSV file.
output = pd.DataFrame({"id": test_data.id, "target": test_preds})
output.to_csv("./working/submission.csv", index=False)