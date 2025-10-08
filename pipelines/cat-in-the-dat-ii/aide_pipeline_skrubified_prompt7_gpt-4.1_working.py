import skrub
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score

# Load the data and create skrub variable for the plan
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Separate target and predictors, mark as y and X
y = data["target"].skb.mark_as_y()
X = data.drop(["target", "id"], axis=1).skb.mark_as_X()

# List of columns by type
binary_cols = [col for col in X.columns if "bin" in col]
ordinal_cols = [col for col in X.columns if "ord" in col]
nominal_cols = [col for col in X.columns if "nom" in col]
cyclical_cols = ["day", "month"]

# Selectors for skrub TableVectorizer
ordinal_selector = skrub.selectors.filter_names(lambda name: "bin" in name or "ord" in name)
low_cardinality_nom_selector = skrub.selectors.filter(lambda col: "nom" in col.name and col.nunique() < 10)
high_cardinality_nom_selector = skrub.selectors.filter(lambda col: "nom" in col.name and col.nunique() >= 10)

# Frequency encoder for high cardinality nominal columns
class FrequencyEncoder:
    def fit(self, X, y=None):
        col = X.columns[0]
        self.freq_ = X[col].value_counts(normalize=True)
        return self
    def transform(self, X):
        col = X.columns[0]
        return pd.DataFrame({col: X[col].map(self.freq_)})

ordinal_encoder = OrdinalEncoder()
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
freq_encoder = FrequencyEncoder()

# TableVectorizer applies the right encoder to each column group
vectorizer = skrub.TableVectorizer(
    specific_transformers=[
        (ordinal_encoder, ordinal_selector),
        (one_hot_encoder, low_cardinality_nom_selector),
        (freq_encoder, high_cardinality_nom_selector)
    ],
)

X_vec = X.skb.apply(vectorizer)

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

# Predict on test
test_data = pd.read_csv("./input/test.csv")
test_preds = learner.predict_proba({"_skrub_X": test_data})[:, 1]

# Save predictions
output = pd.DataFrame({"id": test_data.id, "target": test_preds})
output.to_csv("./working/submission_skrub2.csv", index=False)