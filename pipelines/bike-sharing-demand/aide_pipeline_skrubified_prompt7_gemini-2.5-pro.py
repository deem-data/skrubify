import skrub
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

# Load the data and start the Skrub plan
# Subsample for faster preview computation during plan definition
train_data = pd.read_csv("./input/train.csv")
train_var = skrub.var("train", train_data).skb.subsample(n=1000)

# Separate target (y) and features (X)
# Mark y as the target and X as the entry point for test data
y = train_var["count"].skb.mark_as_y()
X = train_var.drop(["count", "casual", "registered"], axis=1).skb.mark_as_X()

# Apply log1p transformation to the target, this will be used for training
y_log = y.skb.apply_func(np.log1p)

# --- Feature Engineering ---
# The original preprocess_data function is now defined as a series of Skrub operations

# Convert 'datetime' column using apply_func
X_dt = X.assign(datetime=X["datetime"].skb.apply_func(pd.to_datetime))

# Extract time-based features; .dt accessor is tracked by Skrub
X_time_feats = X_dt.assign(
    hour=X_dt["datetime"].dt.hour,
    day_of_week=X_dt["datetime"].dt.dayofweek,
    month=X_dt["datetime"].dt.month,
    year=X_dt["datetime"].dt.year,
    day=X_dt["datetime"].dt.day,
)

# Add interaction feature
X_interact = X_time_feats.assign(
    hour_workingday_interaction=X_time_feats["hour"] * X_time_feats["workingday"]
)

# Add cyclic features. Arithmetic is tracked, but numpy functions need to be wrapped.
X_cyclic = X_interact.assign(
    hour_sin=(X_interact["hour"] * (2.0 * np.pi / 24)).skb.apply_func(np.sin),
    hour_cos=(X_interact["hour"] * (2.0 * np.pi / 24)).skb.apply_func(np.cos),
    day_of_week_sin=(X_interact["day_of_week"] * (2.0 * np.pi / 7)).skb.apply_func(
        np.sin
    ),
    day_of_week_cos=(X_interact["day_of_week"] * (2.0 * np.pi / 7)).skb.apply_func(
        np.cos
    ),
    month_sin=((X_interact["month"] - 1) * (2.0 * np.pi / 12)).skb.apply_func(
        np.sin
    ),
    month_cos=((X_interact["month"] - 1) * (2.0 * np.pi / 12)).skb.apply_func(
        np.cos
    ),
)

# Drop the original datetime column to create the final feature set
X_final = X_cyclic.drop(["datetime"], axis=1)

# --- Model Definition ---
# Define the model to be used
model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)

# Apply the model to the processed features, using the log-transformed target for training
pred_log = X_final.skb.apply(model, y=y_log)

# Reverse the log transformation on the predictions to get the final output
# This completes the end-to-end pipeline definition
pred = pred_log.skb.apply_func(lambda x: x if isinstance(x, LGBMRegressor) else np.expm1(x))

# --- Training and Evaluation ---
# Split the data based on the final output of the plan
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# Create a trainable learner object from the entire plan
learner = pred.skb.make_learner()

# Fit the learner on the training split
learner.fit(splits["train"])

# Predict on the validation split and evaluate
y_pred = learner.predict(splits["test"])
# splits['y_test'] contains the original 'count' values for the validation set
# y_pred contains the final predictions, already transformed back with expm1
rmsle = np.sqrt(mean_squared_log_error(splits["y_test"], y_pred))
print(f"RMSLE with cyclic features: {rmsle}")

# --- Test Prediction and Submission ---
# Load the raw test data
test_data = pd.read_csv("./input/test.csv")

# Predict on the test data. Skrub automatically applies all preprocessing steps.
test_pred = learner.predict({"_skrub_X": test_data})

# Create the submission file
submission = pd.DataFrame(
    {
        "datetime": test_data["datetime"],  # Use the original datetime from test_data
        "count": test_pred,
    }
)
submission.to_csv("./working/submission_skrub.csv", index=False)