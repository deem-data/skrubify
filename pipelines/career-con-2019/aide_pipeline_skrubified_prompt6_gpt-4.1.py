import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
X_train = pd.read_csv("./input/X_train.csv")
y_train = pd.read_csv("./input/y_train.csv")
X_test = pd.read_csv("./input/X_test.csv")

# Start DataOps plan
data = skrub.var("data", X_train).skb.subsample(n=100)
y_data = skrub.var("y_data", y_train)

# Merge sensor data with target variable
merged = data.merge(y_data, on="series_id", how="inner")

# Mark y and X
y = merged["surface"].skb.mark_as_y()
X = merged.drop(["row_id", "series_id", "measurement_number", "group_id", "surface"], axis=1).skb.mark_as_X()

# Normalize features
scaler = StandardScaler()
X_scaled = X.skb.apply(scaler)

# Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
pred = X_scaled.skb.apply(rf, y=y)

# Split
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# Make learner
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Evaluate
y_pred = learner.predict(splits["test"])
accuracy = accuracy_score(splits["y_test"], y_pred)
print(f"Validation Accuracy: {accuracy}")

# Predict on test
test_features = X_test.drop(["row_id", "series_id", "measurement_number"], axis=1)
y_pred_test = learner.predict({"_skrub_X": test_features})

# Save submission
submission = pd.DataFrame({"series_id": X_test["series_id"], "surface": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)