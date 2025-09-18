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
X_feat_eng = X.assign(new_feat=X["feat1"] * X["feat2"])
X_select_feat = X_feat_eng.drop(["id", "feat1", "feat3"], axis=1)

scaler = StandardScaler()
X_scaled = X_select_feat.skb.apply(scaler)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
pred = X_scaled.skb.apply(rf, y=y)

# --- Train/val split ---
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# --- Create learner from last dataop ---
learner = pred.skb.make_learner()

# --- Train ---
learner.fit(splits["train"])

# --- Evaluate ---
y_pred = learner.predict(splits["test"])
acc = accuracy_score(splits["y_test"], y_pred)
print(f"Validation Accuracy: {acc:.4f}")

# --- Predict on test ---
test_data = pd.read_csv("./input/X_test.csv")
y_pred_test = learner.predict({"_skrub_X" : test_data})

# --- Save submission ---
submission = pd.DataFrame({"id": test_data["id"], "label": y_pred_test})
submission.to_csv("./working/submission_skrub.csv", index=False)