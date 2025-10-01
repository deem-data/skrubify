import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load training data
data = pd.read_csv("./input/train.csv")

# Separate features and labels
features = data.drop("label", axis=1)
labels = data["label"]

# Feature engineering
features["new_feat"] = features["feat1"] * features["feat2"]

# Drop unwanted columns
selected_features = features.drop(["id", "feat1", "feat3"], axis=1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(selected_features, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Prepare test data
test_data = pd.read_csv("./input/test.csv")
test_data["new_feat"] = test_data["feat1"] * test_data["feat2"]
X_test = test_data.drop(["id", "feat1", "feat3"], axis=1)
X_test = scaler.transform(X_test)

# Predict and save submission
y_pred_test = rf.predict(X_test)
submission = pd.DataFrame({"id": test_data["id"], "label": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)