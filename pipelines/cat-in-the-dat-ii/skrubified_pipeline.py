import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
import skrub

# --- Load data ---
train_data = skrub.var("train_data", pd.read_csv("./input/train.csv"))
test_data = skrub.var("test_data", pd.read_csv("./input/test.csv"))

# --- Separate target and features ---
y = train_data["target"].skb.mark_as_y()
X = train_data.drop(["target", "id"], axis=1).skb.mark_as_X()
X_test = test_data.drop("id", axis=1)

# --- Identify column types ---
binary_cols = [col for col in X.columns if "bin" in col]
ordinal_cols = [col for col in X.columns if "ord" in col]
nominal_cols = [col for col in X.columns if "nom" in col]
cyclical_cols = ["day", "month"]

# --- Ordinal encoding for binary and ordinal features ---
ordinal_encoder = OrdinalEncoder()
def ordinal_encode(df):
    df = df.copy()
    df[binary_cols + ordinal_cols] = ordinal_encoder.fit_transform(df[binary_cols + ordinal_cols])
    return df
def ordinal_encode_test(df):
    df = df.copy()
    df[binary_cols + ordinal_cols] = ordinal_encoder.transform(df[binary_cols + ordinal_cols])
    return df

X_ord = X.skb.apply(ordinal_encode)
X_test_ord = X_test.skb.apply(ordinal_encode_test)

# --- One-hot encoding for nominal features with low cardinality ---
low_cardinality_nom_cols = [col for col in nominal_cols if X[col].nunique() < 10]
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
def one_hot_encode(df):
    return pd.DataFrame(one_hot_encoder.fit_transform(df[low_cardinality_nom_cols]), index=df.index)
def one_hot_encode_test(df):
    return pd.DataFrame(one_hot_encoder.transform(df[low_cardinality_nom_cols]), index=df.index)

X_low_card_nom = X.skb.apply(one_hot_encode)
X_test_low_card_nom = X_test.skb.apply(one_hot_encode_test)

# --- Frequency encoding for nominal features with high cardinality ---
high_cardinality_nom_cols = [col for col in nominal_cols if X[col].nunique() >= 10]
def freq_encode(df, ref_df):
    df = df.copy()
    for col in high_cardinality_nom_cols:
        freq_encoder = ref_df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq_encoder)
    return df

X_freq = X.skb.apply(lambda df: freq_encode(df, df))
X_test_freq = X_test.skb.apply(lambda df: freq_encode(df, X))

# --- Combine all features ---
def combine_features(X_ord, X_low_card_nom, X_freq):
    # X_ord: ordinal-encoded X
    # X_low_card_nom: one-hot encoded low-cardinality nominals
    # X_freq: frequency-encoded high-cardinality nominals
    # Remove low_cardinality_nom_cols from X_ord, then concat all
    X_comb = pd.concat([X_ord, X_low_card_nom], axis=1)
    X_comb = X_comb.drop(low_cardinality_nom_cols, axis=1)
    # Overwrite high-cardinality nom cols with freq-encoded
    for col in high_cardinality_nom_cols:
        X_comb[col] = X_freq[col]
    return X_comb

X_final = combine_features(X_ord, X_low_card_nom, X_freq)
X_test_final = combine_features(X_test_ord, X_test_low_card_nom, X_test_freq)

# --- Mark as X for Skrub pipeline ---
X_final = X_final.skb.mark_as_X()
X_test_final = X_test_final

# --- Model ---
model = LGBMClassifier()
pred = X_final.skb.apply(model, y=y)

# --- Train/val split ---
splits = pred.skb.train_test_split(train_size=0.8, test_size=0.2, random_state=0)

# --- Create learner ---
learner = pred.skb.make_learner()

# --- Train ---
learner.fit(splits["train"])

# --- Predict on validation set ---
valid_preds = learner.predict(splits["test"], predict_method="predict_proba")[:, 1]

# --- Evaluate ---
roc_auc = roc_auc_score(splits["y_test"], valid_preds)
print(f"Validation ROC AUC Score: {roc_auc}")

# --- Predict on test set ---
test_preds = learner.predict({"_skrub_X": X_test_final}, predict_method="predict_proba")[:, 1]

# --- Save predictions ---
output = pd.DataFrame({"id": test_data["id"], "target": test_preds})
output.to_csv("./working/submission.csv", index=False)