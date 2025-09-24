import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
import skrub

# --- Load data ---
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# --- Separate target and features ---
y = np.log1p(data["count"]).skb.mark_as_y()
X = data.drop("count", axis=1).skb.mark_as_X()

# --- Feature engineering as a sequence of fine-grained steps ---

# Convert datetime to pd.Timestamp
X_dt = X.assign(datetime=pd.to_datetime(X["datetime"]))
X_dt = X.assign(datetime=X["datetime"].skb.apply_func(pd.to_datetime))

# Extract time features
X_dt = X_dt.assign(
    hour=X_dt["datetime"].dt.hour,
    day_of_week=X_dt["datetime"].dt.dayofweek,
    month=X_dt["datetime"].dt.month,
    year=X_dt["datetime"].dt.year,
    day=X_dt["datetime"].dt.day
)

# hour_workingday_interaction
X_dt = X_dt.assign(hour_workingday_interaction=X_dt["hour"] * X_dt["workingday"])

# Cyclic features
X_dt = X_dt.assign(
    hour_sin=np.sin(X_dt["hour"] * (2.0 * np.pi / 24)),
    hour_cos=np.cos(X_dt["hour"] * (2.0 * np.pi / 24)),
    day_of_week_sin=np.sin(X_dt["day_of_week"] * (2.0 * np.pi / 7)),
    day_of_week_cos=np.cos(X_dt["day_of_week"] * (2.0 * np.pi / 7)),
    month_sin=np.sin((X_dt["month"] - 1) * (2.0 * np.pi / 12)),
    month_cos=np.cos((X_dt["month"] - 1) * (2.0 * np.pi / 12))
)

# Drop unwanted columns
def drop_cols(df):
    return df.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")
X_final = X_dt.skb.apply_func(drop_cols)

# --- Model ---
model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
pred = X_final.skb.apply(model, y=y)

# --- Train/val split ---
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# --- Create learner ---
learner = pred.skb.make_learner()

# --- Train ---
learner.fit(splits["train"])

# --- Evaluate ---
y_pred = learner.predict(splits["test"])
# y_val is log1p(count), so we need to expm1 both y_pred and y_val for RMSLE
rmsle = np.sqrt(mean_squared_log_error(np.expm1(splits["y_test"]), np.expm1(y_pred)))
print(f"RMSLE with cyclic features: {rmsle}")

# --- Predict on test ---
test_raw = pd.read_csv("./input/test.csv")

# Feature engineering for test set (must match training pipeline)
test_dt = test_raw.copy()
test_dt["datetime"] = pd.to_datetime(test_dt["datetime"])
test_dt["hour"] = test_dt["datetime"].dt.hour
test_dt["day_of_week"] = test_dt["datetime"].dt.dayofweek
test_dt["month"] = test_dt["datetime"].dt.month
test_dt["year"] = test_dt["datetime"].dt.year
test_dt["day"] = test_dt["datetime"].dt.day
test_dt["hour_workingday_interaction"] = test_dt["hour"] * test_dt["workingday"]
test_dt["hour_sin"] = np.sin(test_dt["hour"] * (2.0 * np.pi / 24))
test_dt["hour_cos"] = np.cos(test_dt["hour"] * (2.0 * np.pi / 24))
test_dt["day_of_week_sin"] = np.sin(test_dt["day_of_week"] * (2.0 * np.pi / 7))
test_dt["day_of_week_cos"] = np.cos(test_dt["day_of_week"] * (2.0 * np.pi / 7))
test_dt["month_sin"] = np.sin((test_dt["month"] - 1) * (2.0 * np.pi / 12))
test_dt["month_cos"] = np.cos((test_dt["month"] - 1) * (2.0 * np.pi / 12))
test_final = test_dt.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

test_pred = learner.predict({"_skrub_X": test_final})

# --- Save submission ---
submission = pd.DataFrame({
    "datetime": test_raw["datetime"],
    "count": np.expm1(test_pred)
})
submission.to_csv("./working/submission.csv", index=False)