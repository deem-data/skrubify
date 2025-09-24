import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
import skrub

# --- Load data ---
train = skrub.var("train", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# --- Feature engineering as a function ---
def feature_engineering(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["day"] = df["datetime"].dt.day
    df["hour_workingday_interaction"] = df["hour"] * df["workingday"]
    df["hour_sin"] = np.sin(df.hour * (2.0 * np.pi / 24))
    df["hour_cos"] = np.cos(df.hour * (2.0 * np.pi / 24))
    df["day_of_week_sin"] = np.sin(df.day_of_week * (2.0 * np.pi / 7))
    df["day_of_week_cos"] = np.cos(df.day_of_week * (2.0 * np.pi / 7))
    df["month_sin"] = np.sin((df.month - 1) * (2.0 * np.pi / 12))
    df["month_cos"] = np.cos((df.month - 1) * (2.0 * np.pi / 12))
    return df.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

# --- Mark y and X ---
y = train["count"].skb.apply_func(np.log1p).skb.mark_as_y()
X = train.skb.apply_func(feature_engineering).drop(["count"], axis=1).skb.mark_as_X()

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
y_val = splits["y_test"]
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred)))
print(f"RMSLE with cyclic features: {rmsle}")

# --- Predict on test ---
test_data = pd.read_csv("./input/test.csv")
test_pred = learner.predict({"_skrub_X": test_data})

# --- Save submission ---
submission = pd.DataFrame({
    "datetime": pd.read_csv("./input/test.csv")["datetime"],
    "count": np.expm1(test_pred),
})
submission.to_csv("./working/submission.csv", index=False)