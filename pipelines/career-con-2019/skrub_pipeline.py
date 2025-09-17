import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import skrub

# Load the data
X_train = skrub.var("X_train", pd.read_csv("./input/X_train.csv"))
y_train = skrub.var("y_train", pd.read_csv("./input/y_train.csv"))

# Merge the sensor data with the target variable
train_data = X_train.merge(y_train, on="series_id", how="inner").skb.subsample(n=256)

# Drop non-feature columns
features = train_data.drop(
    ["surface"], axis=1
).skb.mark_as_X()
features = features.drop(["row_id", "series_id", "measurement_number", "group_id"], axis=1, errors="ignore")
labels = train_data["surface"].skb.mark_as_y()

# Normalize the feature data
scaler = StandardScaler()
features_scaled = features.skb.apply(scaler)

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Apply classifier
preds = features_scaled.skb.apply(rf, y=labels)

# Make learner
learner = preds.skb.make_learner()

# Split the data
splits = preds.skb.train_test_split(test_size=0.2, random_state=42)

# Train and validate
learner.fit(splits["train"])
y_pred = learner.predict(splits["test"])
accuracy = accuracy_score(splits["y_test"], y_pred)
print(f"Validation Accuracy: {accuracy}")

# Prepare the test data
X_test = pd.read_csv("./input/X_test.csv")

# Predict on the test set
test_predictions = learner.predict({"_skrub_X": X_test})

# Save the predictions to a CSV file
submission = pd.DataFrame(
    {"series_id": X_test["series_id"], "surface": test_predictions}
)

submission.to_csv("./working/submission_skrub.csv", index=False)