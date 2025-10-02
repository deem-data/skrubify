import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

import skrub

# --- Load data ---
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Separate labels
y = data["label"].skb.mark_as_y()
X = data.drop("label", axis=1).skb.mark_as_X()

# --- Feature engineering function ---
bin_ord_selector = skrub.selector.filter_names(lambda name: "bin" in name or "ord" in name)
nom_selector = skrub.selector.filter_names(lambda name: "nom" in name)
# num_selector = skrub.selector.filter_names(lambda name: "nom" in name)
cyclical_cols = ["day", "month"]
X_bin = X.skb.select(bin_ord_selector)

# Ordinal encoding for binary and ordinal features
ordinal_encoder = OrdinalEncoder()
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

vectorizer = skrub.TableVectorizer(
    specific_transformers={ordinal_encoder: bin_ord_selector},
    low_cardinality=one_hot_encoder,
    cardinality_threshold=10,
)
