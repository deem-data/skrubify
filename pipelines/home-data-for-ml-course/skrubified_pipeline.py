import skrub
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data = skrub.var("data", pd.read_csv("./input/train.csv"))

# Separate target from predictors
y = data["SalePrice"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log)
X = data.drop(["SalePrice"], axis=1).skb.mark_as_X()

# Preprocessing for numerical
non_obj_selector = skrub.selectors.filter(lambda col: col.dtype != "object")
num_imputer = SimpleImputer(strategy="median")

numeric_cols = X.skb.select(non_obj_selector)
X_num = numeric_cols.skb.apply(num_imputer)

# Preprocessing for categorical
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
cat_imputer = SimpleImputer(strategy="most_frequent")
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = X.skb.select(obj_selector)
X_cat_imputed = X_cat.skb.apply(cat_imputer)
X_cat_vec =  X_cat_imputed.skb.apply(encoder)

# bind together
X_vec = X_num.skb.concat([X_cat_vec], axis=1)

model = GradientBoostingRegressor(random_state=42)
pred_log = X_vec.skb.apply(model, y=y_log)
pred = pred_log.skb.apply_func(lambda x: x if isinstance(x, GradientBoostingRegressor) else np.exp(x))

splits = pred.skb.train_test_split(test_size=0.2, random_state=0)
learner = pred.skb.make_learner()
learner.fit(splits["train"])

y_pred = learner.predict(splits["test"])
score = np.sqrt(mean_squared_error(splits["y_test"], y_pred))
print("RMSE:", score)

test_data = pd.read_csv("./input/test.csv")
test_pred = learner.predict({"_skrub_X": test_data})

output = pd.DataFrame({"Id": test_data.Id, "SalePrice": test_pred})
output.to_csv("./working/submission_skrub.csv", index=False)