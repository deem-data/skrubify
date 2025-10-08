import skrub
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Load the data and create skrub variable for the plan
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Mark target and features
y = data["target"].skb.mark_as_y()
X = data.drop(["target", "id"], axis=1).skb.mark_as_X()

# Selectors for column types
binary_selector = skrub.selectors.filter_names(lambda name: "bin" in name)
ordinal_selector = skrub.selectors.filter_names(lambda name: "ord" in name)
nominal_selector = skrub.selectors.filter_names(lambda name: "nom" in name)

# Select columns
binary_cols = X.skb.select(binary_selector)
ordinal_cols = X.skb.select(ordinal_selector)
nominal_cols = X.skb.select(nominal_selector)

# Ordinal encoding for binary and ordinal features
ord_enc = OrdinalEncoder()
ord_cols = binary_cols.skb.concat([ordinal_cols], axis=1)
ord_cols_enc = ord_cols.skb.apply(ord_enc)

# One-hot encoding for nominal features with low cardinality
low_card_selector = skrub.selectors.filter(lambda col: col.nunique() < 10)
low_card_nom_cols = nominal_cols.skb.select(low_card_selector)
onehot_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
low_card_nom_vec = low_card_nom_cols.skb.apply(onehot_enc)

# Frequency encoding for nominal features with high cardinality
high_card_selector = skrub.selectors.filter(lambda col: col.nunique() >= 10)
high_card_nom_cols = nominal_cols.skb.select(high_card_selector)

from sklearn.base import BaseEstimator, TransformerMixin
class FreqEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        col = X.columns[0]
        self.freq_ = X[col].value_counts(normalize=True)
        return self
    def transform(self, X):
        col = X.columns[0]
        return pd.DataFrame({col: X[col].map(self.freq_)})

high_card_nom_vec = high_card_nom_cols.skb.apply(FreqEncoder())

# Select remaining columns (not encoded above)
other_cols = X.drop(
    binary_cols.columns.tolist() +
    ordinal_cols.columns.tolist() +
    nominal_cols.columns.tolist(),
    axis=1
)

# Combine all features
X_vec = ord_cols_enc.skb.concat([low_card_nom_vec, high_card_nom_vec, other_cols], axis=1)

# Model
model = LGBMClassifier(random_state=42)
pred = X_vec.skb.apply(model, y=y)

# Split for validation
splits = pred.skb.train_test_split(train_size=0.8, test_size=0.2, random_state=0)
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Predict on validation
valid_preds = learner.predict_proba(splits["test"])[:, 1]
roc_auc = roc_auc_score(splits["y_test"], valid_preds)
print(f"Validation ROC AUC Score: {roc_auc}")

# Predict on test set
test_data = pd.read_csv("./input/test.csv")
test_preds = learner.predict_proba({"_skrub_X": test_data.drop("id", axis=1)})[:, 1]

# Save predictions
output = pd.DataFrame({"id": test_data.id, "target": test_preds})
output.to_csv("./working/submission_skrub.csv", index=False)