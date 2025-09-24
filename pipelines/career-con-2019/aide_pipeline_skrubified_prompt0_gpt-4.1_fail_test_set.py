import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import skrub

# --- Load data ---
X_train_var = skrub.var("X_train", pd.read_csv("./input/X_train.csv")).skb.subsample(n=100)
y_train_var = skrub.var("y_train", pd.read_csv("./input/y_train.csv")).skb.subsample(n=100)
X_test_df = pd.read_csv("./input/X_test.csv")

# --- Merge sensor data with target variable ---
train_data = X_train_var.merge(y_train_var, on="series_id", how="inner")

# --- Drop non-feature columns and mark features/labels ---
features = train_data.drop(
    ["row_id", "series_id", "measurement_number", "group_id", "surface"], axis=1
).skb.mark_as_X()
labels = train_data["surface"].skb.mark_as_y()

# --- Normalize the feature data ---
scaler = StandardScaler()
features_scaled = features.skb.apply(scaler)

# --- Model ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
pred = features_scaled.skb.apply(rf, y=labels)

# --- Train/val split ---
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# --- Create learner ---
learner = pred.skb.make_learner()

# --- Train ---
learner.fit(splits["train"])

# --- Evaluate ---
y_pred = learner.predict(splits["test"])
acc = accuracy_score(splits["y_test"], y_pred)
print(f"Validation Accuracy: {acc}")

# --- Prepare test data ---
test_features = X_test_df.drop(["row_id", "series_id", "measurement_number"], axis=1)
test_features_scaled = scaler.transform(test_features)

# --- Predict on test set ---
test_predictions = learner.predict({"_skrub_X": test_features})

# --- Save predictions ---
submission = pd.DataFrame(
    {"series_id": X_test_df["series_id"], "surface": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)