import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

# Load data
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Mark y and X
y = data["count"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log1p)
X = data.drop(["count"], axis=1).skb.mark_as_X()

# Feature engineering steps, all as fine-grained skrub operations
X_dt = X.assign(datetime=X["datetime"].skb.apply_func(pd.to_datetime))
X_hour = X_dt.assign(hour=X_dt["datetime"].dt.hour)
X_day_of_week = X_hour.assign(day_of_week=X_hour["datetime"].dt.dayofweek)
X_month = X_day_of_week.assign(month=X_day_of_week["datetime"].dt.month)
X_year = X_month.assign(year=X_month["datetime"].dt.year)
X_day = X_year.assign(day=X_year["datetime"].dt.day)
X_hour_workingday = X_day.assign(hour_workingday_interaction=X_day["hour"] * X_day["workingday"])

# Cyclic features
X_hour_sin = X_hour_workingday.assign(hour_sin=X_hour_workingday["hour"].skb.apply_func(lambda h: np.sin(h * (2.0 * np.pi / 24))))
X_hour_cos = X_hour_sin.assign(hour_cos=X_hour_sin["hour"].skb.apply_func(lambda h: np.cos(h * (2.0 * np.pi / 24))))
X_dow_sin = X_hour_cos.assign(day_of_week_sin=X_hour_cos["day_of_week"].skb.apply_func(lambda d: np.sin(d * (2.0 * np.pi / 7))))
X_dow_cos = X_dow_sin.assign(day_of_week_cos=X_dow_sin["day_of_week"].skb.apply_func(lambda d: np.cos(d * (2.0 * np.pi / 7))))
X_month_sin = X_dow_cos.assign(month_sin=X_dow_cos["month"].skb.apply_func(lambda m: np.sin((m - 1) * (2.0 * np.pi / 12))))
X_month_cos = X_month_sin.assign(month_cos=X_month_sin["month"].skb.apply_func(lambda m: np.cos((m - 1) * (2.0 * np.pi / 12))))

# Drop columns
X_final = X_month_cos.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

# Model
model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
pred_log = X_final.skb.apply(model, y=y_log)
pred = pred_log.skb.apply_func(np.expm1)

# Split
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Evaluate
y_pred = learner.predict(splits["test"])
rmsle = np.sqrt(mean_squared_log_error(splits["y_test"], y_pred))
print(f"RMSLE with cyclic features: {rmsle}")

# Test prediction
test_data = pd.read_csv("./input/test.csv")
test_pred = learner.predict({"_skrub_X": test_data})

# Prepare submission
submission = pd.DataFrame({
    "datetime": pd.read_csv("./input/test.csv")["datetime"],
    "count": test_pred
})
submission.to_csv("./working/submission_skrub.csv", index=False)