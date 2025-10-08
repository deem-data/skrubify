import skrub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data and create skrub variable for the plan
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Mark y and X
y = data["Cover_Type"].skb.mark_as_y()
X = data.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X()

# Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
pred = X.skb.apply(rf, y=y)

# Split for validation
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# Create learner and fit
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Validate
y_pred = learner.predict(splits["test"])
accuracy = accuracy_score(splits["y_test"], y_pred)
print(f"Validation Accuracy: {accuracy}")

# Predict on test data
test_data = pd.read_csv("./input/test.csv")
test_ids = test_data["Id"]
test_features = test_data.drop("Id", axis=1)
test_predictions = learner.predict({"_skrub_X": test_features})

# Save the predictions
submission = pd.DataFrame({"Id": test_ids, "Cover_Type": test_predictions})
submission.to_csv("./working/submission_skrub.csv", index=False)