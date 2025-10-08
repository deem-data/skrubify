import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# --- Skrub Pipeline Definition ---

# Load and merge training data to create the initial dataset for the plan
X_train_df = pd.read_csv("./input/X_train.csv")
y_train_df = pd.read_csv("./input/y_train.csv")
train_data = X_train_df.merge(y_train_df, on="series_id", how="inner")

# DataOps plan always begins with a variable, here the function tracking starts
data_var = skrub.var("data", train_data)
# Subsampling for faster preview computation, this step is automatically skipped in the final pipeline
data_var = data_var.skb.subsample(n=100)

# Add the operation for separating features / labels
y = data_var["surface"].skb.mark_as_y()

# Mark the intermediate that will be the entry point for the test data.
# The schema at this point must match the raw test data schema (X_test.csv).
# The merged training data has 'group_id' and 'surface', which are not in the test data, so we drop them first.
X = data_var.drop(["group_id", "surface"], axis=1).skb.mark_as_X()

# Add feature engineering operations to the plan
# This operation will be applied to both train and test data automatically
X_features = X.drop(["row_id", "series_id", "measurement_number"], axis=1)

# Add scaling operation to the plan
scaler = StandardScaler()
X_scaled = X_features.skb.apply(scaler)

# Add the model to the plan
rf = RandomForestClassifier(n_estimators=100, random_state=42)
pred = X_scaled.skb.apply(rf, y=y)
# Pipeline definition is finished here

# --- Pipeline Execution ---

# Get train and validation data by calling this method on the last DataOp in the pipeline
# Skrub automatically computes the appropriate X and y from the DataOps plan for each split
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# Create a trainable learner object from the entire pipeline plan
learner = pred.skb.make_learner()

# Pass the train data to the learner; the data is injected and processed according to the plan
learner.fit(splits["train"])

# Evaluate the pipeline on the validation set
y_pred = learner.predict(splits["test"])
accuracy = accuracy_score(splits["y_test"], y_pred)
print(f"Validation Accuracy: {accuracy}")

# --- Prediction on Test Data ---

# Load the raw test data
X_test = pd.read_csv("./input/X_test.csv")

# Predict on the test data by injecting it at the intermediate marked as X.
# Skrub handles all subsequent preprocessing steps (dropping columns, scaling) automatically.
test_predictions = learner.predict({"_skrub_X": X_test})

# --- Save Submission ---

# Save the predictions to a CSV file
submission = pd.DataFrame(
    {"series_id": X_test["series_id"], "surface": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)