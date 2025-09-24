import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
import skrub

# --- Load data ---
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# --- Define y and X ---
y = np.log1p(data["count"]).skb.mark_as_y()
X = data.drop(["count"], axis=1).skb.mark_as_X()

# --- Feature engineering (fine-grained, no UDFs) ---
# X_dt = X.assign(datetime=pd.to_datetime(X["datetime"]))
X_dt = X.assign(datetime=X["datetime"].skb.apply_func(pd.to_datetime))
X_hour = X_dt.assign(hour=X_dt["datetime"].dt.hour)
X_day_of_week = X_hour.assign(day_of_week=X_hour["datetime"].dt.dayofweek)
X_month = X_day_of_week.assign(month=X_day_of_week["datetime"].dt.month)
X_year = X_month.assign(year=X_month["datetime"].dt.year)
X_day = X_year.assign(day=X_year["datetime"].dt.day)
X_hour_workingday = X_day.assign(hour_workingday_interaction=X_day["hour"] * X_day["workingday"])

# Cyclic features
X_hour_sin = X_hour_workingday.assign(hour_sin=np.sin(X_hour_workingday["hour"] * (2.0 * np.pi / 24)))
X_hour_cos = X_hour_sin.assign(hour_cos=np.cos(X_hour_sin["hour"] * (2.0 * np.pi / 24)))
X_dow_sin = X_hour_cos.assign(day_of_week_sin=np.sin(X_hour_cos["day_of_week"] * (2.0 * np.pi / 7)))
X_dow_cos = X_dow_sin.assign(day_of_week_cos=np.cos(X_dow_sin["day_of_week"] * (2.0 * np.pi / 7)))
X_month_sin = X_dow_cos.assign(month_sin=np.sin((X_dow_cos["month"] - 1) * (2.0 * np.pi / 12)))
X_month_cos = X_month_sin.assign(month_cos=np.cos((X_month_sin["month"] - 1) * (2.0 * np.pi / 12)))

# Drop unwanted columns
X_final = X_month_cos.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

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
# No explicit feature engineering needed, handled by learner
test_pred = learner.predict({"_skrub_X": test_raw})
submission = pd.DataFrame({
    "datetime": test_raw["datetime"],
    "count": np.expm1(test_pred)
})
submission.to_csv("./working/submission_skrub.csv", index=False)