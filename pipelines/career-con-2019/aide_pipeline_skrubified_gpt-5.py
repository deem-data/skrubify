import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import skrub

# --- Load data ---
X_train_df = pd.read_csv("./input/X_train.csv")
y_train_df = pd.read_csv("./input/y_train.csv")

X_train_var = skrub.var("X_train", X_train_df).skb.subsample(n=100)
y_train_var = skrub.var("y_train", y_train_df).skb.subsample(n=100)

# --- Merge and define X/y ---
merged = X_train_var.merge(y_train_var, on="series_id", how="inner")
y = merged["surface"].skb.mark_as_y()
X = merged.skb.mark_as_X()

# --- Feature selection: drop non-feature columns (robust to missing columns) ---
def drop_non_feature_columns(df):
    cols_to_drop = ["row_id", "series_id", "measurement_number", "group_id", "surface"]
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)

X_features = X.skb.apply_func(drop_non_feature_columns)

# --- Scaling ---
scaler = StandardScaler()
X_scaled = X_features.skb.apply(scaler)

# --- Model ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
pred = X_scaled.skb.apply(rf, y=y)

# --- Train/val split ---
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# --- Learner ---
learner = pred.skb.make_learner()

# --- Train ---
learner.fit(splits["train"])

# --- Evaluate ---
y_pred = learner.predict(splits["test"])
acc = accuracy_score(splits["y_test"], y_pred)
print(f"Validation Accuracy: {acc}")

# --- Predict on test ---
X_test = pd.read_csv("./input/X_test.csv")
test_predictions = learner.predict({"_skrub_X": X_test})

# --- Save submission ---
submission = pd.DataFrame({"series_id": X_test["series_id"], "surface": test_predictions})
submission.to_csv("./working/submission.csv", index=False)