import skrub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the data
X_train_df = pd.read_csv("./input/X_train.csv")
y_train_df = pd.read_csv("./input/y_train.csv")
X_test_df = pd.read_csv("./input/X_test.csv")

# Start DataOps plan
data_X = skrub.var("X_train", X_train_df).skb.subsample(n=100)
data_y = skrub.var("y_train", y_train_df).skb.subsample(n=100)

# Merge sensor data with target variable
train_data = data_X.merge(data_y, on="series_id", how="inner")

# Mark y and X
y = train_data["surface"].skb.mark_as_y()
X = train_data.drop(
    ["row_id", "series_id", "measurement_number", "group_id", "surface"], axis=1
).skb.mark_as_X()

# Normalize the feature data
scaler = StandardScaler()
X_scaled = X.skb.apply(scaler)

# Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
pred = X_scaled.skb.apply(rf, y=y)

# Split for validation
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# Create learner
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Evaluate
y_pred = learner.predict(splits["test"])
acc = accuracy_score(splits["y_test"], y_pred)
print(f"Validation Accuracy: {acc}")

# Predict on test set
test_features = X_test_df.drop(["row_id", "series_id", "measurement_number"], axis=1)
test_pred = learner.predict({"_skrub_X": test_features})

# Save predictions
submission = pd.DataFrame(
    {"series_id": X_test_df["series_id"], "surface": test_pred}
)
submission.to_csv("./working/submission_skrub.csv", index=False)