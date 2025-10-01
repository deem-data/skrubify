import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import skrub

# --- Load data ---
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100) # subsampling for faster preview

# Separate labels
y = data["label"].skb.mark_as_y()
X = data.drop("label", axis=1).skb.mark_as_X()

# --- Feature engineering function ---
bin_selector = skrub.selector.filter_names(lambda name: "bin" in name or  "nom" in name)
ord_selector = skrub.selector.filter_names(lambda name: "ord" in name)
num_selector = skrub.selector.filter_names(lambda name: "nom" in name)
cyclical_cols = ["day", "month"]
X_bin = X.skb.select(bin_selector)

# Ordinal encoding for binary and ordinal features
ordinal_encoder = OrdinalEncoder()
X[binary_cols + ordinal_cols] = ordinal_encoder.fit_transform(
    X[binary_cols + ordinal_cols]
)

vectorizer = skrub.TableVectorizer(specific_transformers={
    ord_selector:
})
