import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
import skrub

# --- Load data ---
train = skrub.var("train", pd.read_csv("./input/train.csv")).skb.subsample(n=100)
test = skrub.var("test", pd.read_csv("./input/test.csv")).skb.subsample(n=100)

# --- Feature engineering steps as fine-grained skrub DataOps ---

# 1. Parse datetime
train_dt = train.assign(datetime=pd.to_datetime(train["datetime"]))
test_dt = test.assign(datetime=pd.to_datetime(test["datetime"]))

# 2. Extract datetime features
def extract_datetime_features(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["day"] = df["datetime"].dt.day
    return df

train_dt_feat = train_dt.skb.apply_func(extract_datetime_features)
test_dt_feat = test_dt.skb.apply_func(extract_datetime_features)

# 3. hour_workingday_interaction
train_inter = train_dt_feat.assign(hour_workingday_interaction=train_dt_feat["hour"] * train_dt_feat["workingday"])
test_inter = test_dt_feat.assign(hour_workingday_interaction=test_dt_feat["hour"] * test_dt_feat["workingday"])

# 4. Add cyclic features
def add_cyclic_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(df["hour"] * (2.0 * np.pi / 24))
    df["hour_cos"] = np.cos(df["hour"] * (2.0 * np.pi / 24))
    df["day_of_week_sin"] = np.sin(df["day_of_week"] * (2.0 * np.pi / 7))
    df["day_of_week_cos"] = np.cos(df["day_of_week"] * (2.0 * np.pi / 7))
    df["month_sin"] = np.sin((df["month"] - 1) * (2.0 * np.pi / 12))
    df["month_cos"] = np.cos((df["month"] - 1) * (2.0 * np.pi / 12))
    return df

train_cyclic = train_inter.skb.apply_func(add_cyclic_features)
test_cyclic = test_inter.skb.apply_func(add_cyclic_features)

# 5. Drop unwanted columns
def drop_columns(df):
    return df.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

train_final = train_cyclic.skb.apply_func(drop_columns)
test_final = test_cyclic.skb.apply_func(drop_columns)

# --- Separate features and target ---
y = train_final["count"].skb.apply_func(np.log1p).skb.mark_as_y()
X = train_final.drop(["count"], axis=1).skb.mark_as_X()

# --- Model ---
model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
pred = X.skb.apply(model, y=y)

# --- Train/val split ---
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# --- Learner ---
learner = pred.skb.make_learner()

# --- Train ---
learner.fit(splits["train"])

# --- Evaluate ---
y_val_pred = learner.predict(splits["test"])
y_val_true = np.expm1(splits["y_test"])
y_val_pred_exp = np.expm1(y_val_pred)
rmsle = np.sqrt(mean_squared_log_error(y_val_true, y_val_pred_exp))
print(f"RMSLE with cyclic features: {rmsle}")

# --- Predict on test ---
test_pred_log = learner.predict({"_skrub_X": test_final})
test_pred = np.expm1(test_pred_log)

# --- Save submission ---
test_datetime = pd.read_csv("./input/test.csv")["datetime"]
submission = pd.DataFrame({"datetime": test_datetime, "count": test_pred})
submission.to_csv("./working/submission.csv", index=False)