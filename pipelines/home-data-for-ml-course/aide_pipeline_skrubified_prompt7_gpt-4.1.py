import skrub
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Start skrub DataOps plan
data = skrub.var("data", train_data).skb.subsample(n=100)

# Separate target from predictors
y = data["SalePrice"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log)
X = data.drop(["SalePrice"], axis=1).skb.mark_as_X()

# Selectors for column types
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")
cat_selector = skrub.selectors.filter(lambda col: col.dtype == "object")

# Numerical preprocessing
num_imputer = SimpleImputer(strategy="median")
X_num = X.skb.select(num_selector)
X_num_imputed = X_num.skb.apply(num_imputer)

# Categorical preprocessing
cat_imputer = SimpleImputer(strategy="most_frequent")
cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = X.skb.select(cat_selector)
X_cat_imputed = X_cat.skb.apply(cat_imputer)
X_cat_encoded = X_cat_imputed.skb.apply(cat_encoder)

# Concatenate processed features
X_processed = X_num_imputed.skb.concat([X_cat_encoded], axis=1)

# Model
model = GradientBoostingRegressor(random_state=42)
pred_log = X_processed.skb.apply(model, y=y_log)
pred = pred_log.skb.apply_func(lambda x, mode: np.exp(x) if mode == "transform" or mode mode == "predict" else x, skrub.eval_mode())

# Split data
splits = pred.skb.train_test_split(test_size=0.2, random_state=0)

# Create learner
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Predict on validation
y_pred = learner.predict(splits["test"])
rmse = np.sqrt(mean_squared_error(splits["y_test"], y_pred))
print("RMSE:", rmse)

# Predict on test data
test_preds = learner.predict({"_skrub_X": test_data})

# Save test predictions to file
output = pd.DataFrame({"Id": test_data.Id, "SalePrice": test_preds})
output.to_csv("./working/submission_skrub.csv", index=False)
