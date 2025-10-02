import skrub
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

# Load data and create skrub variable with subsample for preview
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Feature engineering steps as fine-grained DataOps
data_dt = data.assign(datetime=data["datetime"].skb.apply_func(pd.to_datetime))
hour = data_dt["datetime"].dt.hour
day_of_week = data_dt["datetime"].dt.dayofweek
month = data_dt["datetime"].dt.month
year = data_dt["datetime"].dt.year
day = data_dt["datetime"].dt.day
hour_workingday_interaction = hour * data_dt["workingday"]

hour_sin = hour.skb.apply_func(lambda x: np.sin(x * (2.0 * np.pi / 24)))
hour_cos = hour.skb.apply_func(lambda x: np.cos(x * (2.0 * np.pi / 24)))
day_of_week_sin = day_of_week.skb.apply_func(lambda x: np.sin(x * (2.0 * np.pi / 7)))
day_of_week_cos = day_of_week.skb.apply_func(lambda x: np.cos(x * (2.0 * np.pi / 7)))
month_sin = month.skb.apply_func(lambda x: np.sin((x - 1) * (2.0 * np.pi / 12)))
month_cos = month.skb.apply_func(lambda x: np.cos((x - 1) * (2.0 * np.pi / 12)))

# Assign all engineered features
X_feat = data_dt.assign(
    hour=hour,
    day_of_week=day_of_week,
    month=month,
    year=year,
    day=day,
    hour_workingday_interaction=hour_workingday_interaction,
    hour_sin=hour_sin,
    hour_cos=hour_cos,
    day_of_week_sin=day_of_week_sin,
    day_of_week_cos=day_of_week_cos,
    month_sin=month_sin,
    month_cos=month_cos
)

# Drop columns not used for training
X_drop = X_feat.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

# Mark y and X
y = X_drop["count"].skb.mark_as_y().skb.apply_func(np.log1p)
X = X_drop.drop(["count"], axis=1).skb.mark_as_X()

# Model
model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
pred_log = X.skb.apply(model, y=y)
pred = pred_log.skb.apply_func(np.expm1)

# Split for validation
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Evaluate
y_pred = learner.predict(splits["test"])
rmsle = np.sqrt(mean_squared_log_error(splits["y_test"], y_pred))
print(f"RMSLE with cyclic features: {rmsle}")

# Test prediction
test_data = pd.read_csv("./input/test.csv")
test_dt = test_data.assign(datetime=test_data["datetime"].apply(pd.to_datetime))
hour = test_dt["datetime"].dt.hour
day_of_week = test_dt["datetime"].dt.dayofweek
month = test_dt["datetime"].dt.month
year = test_dt["datetime"].dt.year
day = test_dt["datetime"].dt.day
hour_workingday_interaction = hour * test_dt["workingday"]
hour_sin = np.sin(hour * (2.0 * np.pi / 24))
hour_cos = np.cos(hour * (2.0 * np.pi / 24))
day_of_week_sin = np.sin(day_of_week * (2.0 * np.pi / 7))
day_of_week_cos = np.cos(day_of_week * (2.0 * np.pi / 7))
month_sin = np.sin((month - 1) * (2.0 * np.pi / 12))
month_cos = np.cos((month - 1) * (2.0 * np.pi / 12))
test_feat = test_dt.assign(
    hour=hour,
    day_of_week=day_of_week,
    month=month,
    year=year,
    day=day,
    hour_workingday_interaction=hour_workingday_interaction,
    hour_sin=hour_sin,
    hour_cos=hour_cos,
    day_of_week_sin=day_of_week_sin,
    day_of_week_cos=day_of_week_cos,
    month_sin=month_sin,
    month_cos=month_cos
)
test_X = test_feat.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

# Predict using the pipeline
y_pred_test = learner.predict({"_skrub_X": test_X})

# Prepare submission
submission = pd.DataFrame({
    "datetime": test_data["datetime"],
    "count": y_pred_test
})
submission.to_csv("./working/submission.csv", index=False)