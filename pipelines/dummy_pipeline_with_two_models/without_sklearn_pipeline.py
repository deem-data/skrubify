import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load training data
data = pd.read_csv("./input/train.csv")

# Separate features and labels
features = data.drop(["label1","label2"], axis=1)
labels = data[["label1","label2"]]

# Split data
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train models
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(X_train, y_train["label1"])

model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train, y_train["label2"])

# Evaluate
y_pred1 = model1.predict(X_val)
y_pred2 = model2.predict(X_val)
y_pred = np.column_stack((y_pred1, y_pred2))
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Prepare test data
test_data = pd.read_csv("./input/test.csv")
X_test = scaler.transform(test_data)

# Predict and save submission
y_pred_test1 = model1.predict(X_test)
y_pred_test2 = model2.predict(X_test)
submission = pd.DataFrame({"id": test_data["id"], "label1": y_pred_test1, "label2": y_pred_test2})
submission.to_csv("./working/submission.csv", index=False)