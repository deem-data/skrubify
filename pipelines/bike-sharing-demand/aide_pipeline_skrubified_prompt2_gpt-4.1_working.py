import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

# Load the data
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Mark y and X
y = data["count"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log1p)
X = data.drop(["count"], axis=1).skb.mark_as_X()

# Feature engineering steps
X_dt = X.assign(datetime=X["datetime"].skb.apply_func(pd.to_datetime))
X_hour = X_dt.assign(hour=X_dt["datetime"].dt.hour)
X_day_of_week = X_hour.assign(day_of_week=X_hour["datetime"].dt.dayofweek)
X_month = X_day_of_week.assign(month=X_day_of_week["datetime"].dt.month)
X_year = X_month.assign(year=X_month["datetime"].dt.year)
X_day = X_year.assign(day=X_year["datetime"].dt.day)
X_hour_workingday = X_day.assign(hour_workingday_interaction=X_day["hour"] * X_day["workingday"])

# Cyclic features
X_hour_sin = X_hour_workingday.assign(hour_sin=X_hour_workingday["hour"].skb.apply_func(lambda x: np.sin(x * (2.0 * np.pi / 24))))
X_hour_cos = X_hour_sin.assign(hour_cos=X_hour_sin["hour"].skb.apply_func(lambda x: np.cos(x * (2.0 * np.pi / 24))))
X_dow_sin = X_hour_cos.assign(day_of_week_sin=X_hour_cos["day_of_week"].skb.apply_func(lambda x: np.sin(x * (2.0 * np.pi / 7))))
X_dow_cos = X_dow_sin.assign(day_of_week_cos=X_dow_sin["day_of_week"].skb.apply_func(lambda x: np.cos(x * (2.0 * np.pi / 7))))
X_month_sin = X_dow_cos.assign(month_sin=X_dow_cos["month"].skb.apply_func(lambda x: np.sin((x - 1) * (2.0 * np.pi / 12))))
X_month_cos = X_month_sin.assign(month_cos=X_month_sin["month"].skb.apply_func(lambda x: np.cos((x - 1) * (2.0 * np.pi / 12))))

# Drop columns
X_final = X_month_cos.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

# Model
model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
pred = X_final.skb.apply(model, y=y_log)

# Split for validation
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# Create learner
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Predict and evaluate
y_pred = learner.predict(splits["test"])
rmsle = np.sqrt(mean_squared_log_error(np.expm1(splits["y_test"]), np.expm1(y_pred)))
print(f"RMSLE with cyclic features: {rmsle}")

# Prepare submission
test_data = pd.read_csv("./input/test.csv")
y_pred_test = learner.predict({"_skrub_X": test_data})
submission = pd.DataFrame({
    "datetime": test_data["datetime"],
    "count": np.expm1(y_pred_test)
})
submission.to_csv("./working/submission_skrub.csv", index=False)