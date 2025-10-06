import numpy as np
import skrub
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._set_output import _wrap_data_with_container
from sklearn.utils.validation import check_is_fitted

df = pd.read_csv('../cat-in-the-dat-ii/input/train.csv')

data = skrub.var("data", df).skb.subsample(n=len(df)// 10)
X = data.drop(["target","id"], axis=1).skb.mark_as_X()
y = data["target"].skb.mark_as_y()

bin_ord_selector = skrub.selectors.filter_names(lambda name: "bin" in name or "ord" in name)
low_cardinality_nom_selector = skrub.selectors.filter(lambda col: "nom" in col.name and col.nunique() < 10)
high_cardinality_nom_selector = skrub.selectors.filter(lambda col: "nom" in col.name and col.nunique() >= 10)

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vocab_ = None

    def fit(self, X: pd.DataFrame, y=None):
        # X is DataFrame with a single column
        self.vocab_ = X.iloc[:,0].value_counts(normalize=True)
        return self

    def transform(self, X):
        return pd.DataFrame({X.columns[0] : X.iloc[:,0].map(self.vocab_)})


ordinal_encoder = OrdinalEncoder()
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
freq_encoder = FrequencyEncoder()

vectorizer = skrub.TableVectorizer(
    specific_transformers=[
        (ordinal_encoder, bin_ord_selector),
        (one_hot_encoder, low_cardinality_nom_selector),
        (freq_encoder, high_cardinality_nom_selector)
    ],
)
model = LGBMClassifier(random_state=42)

vec_X = X.skb.apply(vectorizer)
pred = vec_X.skb.apply(model, y=y)

splits = pred.skb.train_test_split(train_size=0.8, test_size=0.2, random_state=0)
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Predict on the validation set
valid_preds = learner.predict_proba(splits["test"])[:, 1]

# Evaluate the model
roc_auc = roc_auc_score(splits["y_test"], valid_preds)
print(f"Validation ROC AUC Score: {roc_auc}")

# Predict on the test set
test_data = pd.read_csv("./input/test.csv")
ids = test_data["id"]
test_data = test_data.drop(["id"], axis=1)
test_preds = learner.predict_proba({"_skrub_X" : test_data})[:, 1]

# Save the predictions to a CSV file
output = pd.DataFrame({"id": ids, "target": test_preds})
output.to_csv("./working/submission_skrub.csv", index=False)





