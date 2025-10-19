import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# --- Skrub DataOps Pipeline Definition ---

# Load the initial training data sources
# Subsampling for faster preview computation (automatically skipped in final pipeline execution)
X_train_var = skrub.var("X_train", pd.read_csv("./input/X_train.csv")).skb.subsample(n=100)
y_train_var = skrub.var("y_train", pd.read_csv("./input/y_train.csv")).skb.subsample(n=100)

# Merge the sensor data with the target variable within the Skrub plan
merged_data = X_train_var.skb.merge(y_train_var, on="series_id", how="inner")

# Define the label column and mark it as 'y'
y = merged_data["surface"].skb.mark_as_y()

# To ensure consistent feature sets between training and prediction,
# we explicitly define the feature columns.
# This step is typically done once during pipeline design, based on data schema.
# For this example, we infer them from a sample of X_train.csv.
# Non-feature columns in X_train that are not part of the final feature set:
# 'row_id', 'series_id', 'measurement_number', 'group_id'
temp_X_train_sample = pd.read_csv("./input/X_train.csv")
non_feature_cols_in_X_train_df = ["row_id", "series_id", "measurement_number", "group_id"]
feature_cols = [col for col in temp_X_train_sample.columns if col not in non_feature_cols_in_X_train_df]

# Select only the actual feature columns and mark them as 'X'
# This ensures that the input to the scaler and model is always the same set of features,
# whether from merged_data (for training) or test_data_raw (for prediction).
X = merged_data[feature_cols].skb.mark_as_X()

# Add scaling operation to the plan
scaler = StandardScaler()
X_scaled = X.skb.apply(scaler)

# Add the Random Forest classifier to the plan
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# The final prediction step, which will be used to create the learner
pred = X_scaled.skb.apply(rf, y=y)
# Pipeline definition is finished here

# --- Skrub Pipeline Execution ---

# Get train and validation data splits from the defined DataOps plan
# Skrub automatically computes the X and y from the DataOps plan
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# Create a trainable learner object from the pipeline
learner = pred.skb.make_learner()

# Train the learner using the training split
# Skrub handles the full preprocessing chain defined in the plan
learner.fit(splits["train"])

# Evaluate the pipeline on the validation split
y_pred_val = learner.predict(splits["test"])
accuracy = accuracy_score(splits["y_test"], y_pred_val)
print(f"Validation Accuracy: {accuracy}")

# Predict on the test data
test_data_raw = pd.read_csv("./input/X_test.csv")
# Pass the raw test data directly to the learner.
# Skrub will automatically apply all preprocessing steps defined in the pipeline
# (i.e., selecting `feature_cols` and scaling) before making predictions.
# The `_skrub_X` key is used to inject the data at the intermediate marked as X.
y_pred_test = learner.predict({"_skrub_X" : test_data_raw})

# --- Save submission ---
# Use the original series_id from the raw test data for the submission file
submission = pd.DataFrame({"series_id": test_data_raw["series_id"], "surface": y_pred_test})
submission.to_csv("./working/submission_skrub.csv", index=False)