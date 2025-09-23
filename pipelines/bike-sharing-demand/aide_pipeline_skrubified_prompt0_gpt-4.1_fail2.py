import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
import skrub

# --- Load data ---
train = skrub.var("train", pd.read_csv("./input/train.csv")).skb.subsample(n=100)
# test data will be loaded later for prediction

# --- Feature engineering as a sequence of skrub operations ---

# 1. Parse datetime
train_dt = train.assign(datetime=pd.to_datetime(train["datetime"]))

# 2. Extract time features
train_dt = train_dt.assign(
    hour=train_dt["datetime"].dt.hour,
    day_of_week=train_dt["datetime"].dt.dayofweek,
    month=train_dt["datetime"].dt.month,
    year=train_dt["datetime"].dt.year,
    day=train_dt["datetime"].dt.day,
)

# 3. Interaction feature
train_dt = train_dt.assign(
    hour_workingday_interaction=train_dt["hour"] * train_dt["workingday"]
)

# 4. Cyclic features
train_dt = train_dt.assign(
    hour_sin=np.sin(train_dt["hour"] * (2.0 * np.pi / 24)),
    hour_cos=np.cos(train_dt["hour"] * (2.0 * np.pi / 24)),
    day_of_week_sin=np.sin(train_dt["day_of_week"] * (2.0 * np.pi / 7)),
    day_of_week_cos=np.cos(train_dt["day_of_week"] * (2.0 * np.pi / 7)),
    month_sin=np.sin((train_dt["month"] - 1) * (2.0 * np.pi / 12)),
    month_cos=np.cos((train_dt["month"] - 1) * (2.0 * np.pi / 12)),
)

# 5. Drop unwanted columns
X = train_dt.drop(["datetime", "casual", "registered", "count"], axis=1, errors="ignore").skb.mark_as_X()
y = np.log1p(train_dt["count"]).skb.mark_as_y()

# --- Model ---
model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
pred = X.skb.apply(model, y=y)

# --- Train/val split ---
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# --- Create learner ---
learner = pred.skb.make_learner()

# --- Train ---
learner.fit(splits["train"])

# --- Evaluate ---
y_pred = learner.predict(splits["test"])
rmsle = np.sqrt(mean_squared_log_error(np.expm1(splits["y_test"]), np.expm1(y_pred)))
print(f"RMSLE with cyclic features: {rmsle}")

# --- Predict on test set ---
test_raw = pd.read_csv("./input/test.csv")
# Apply the same feature engineering as above
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
X_test = test_dt.drop(["datetime", "casual", "registered", "count"], axis=1, errors="ignore")

test_pred = learner.predict({"_skrub_X": X_test})

# --- Save submission ---
submission = pd.DataFrame(
    {
        "datetime": test_raw["datetime"],
        "count": np.expm1(test_pred),
    }
)
submission.to_csv("./working/submission.csv", index=False)