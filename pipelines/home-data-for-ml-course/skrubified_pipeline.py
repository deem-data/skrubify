import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import skrub

# --- Load data ---
train_data = skrub.var("train_data", pd.read_csv("./input/train.csv"))
test_data = skrub.var("test_data", pd.read_csv("./input/test.csv"))

# --- Separate target and predictors ---
y = train_data["SalePrice"].skb.mark_as_y()
X = train_data.drop(["SalePrice"], axis=1).skb.mark_as_X()

# --- Identify columns ---
num_cols = X.select_dtypes(exclude=["object"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# --- Preprocessing transformers ---
numerical_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# --- Model ---
model = GradientBoostingRegressor()

# --- Pipeline: log-transform target, preprocess, fit model ---
def log_transform_y(y):
    return np.log(y)

def exp_transform_preds(preds):
    return np.exp(preds)

y_log = y.skb.apply(log_transform_y)
X_preprocessed = X.skb.apply(preprocessor)
pred = X_preprocessed.skb.apply(model, y=y_log)

# --- Train/val split ---
splits = pred.skb.train_test_split(test_size=0.2, random_state=0)

# --- Learner ---
learner = pred.skb.make_learner()

# --- Train ---
learner.fit(splits["train"])

# --- Predict and evaluate on validation ---
val_preds_log = learner.predict(splits["test"])
rmse = mean_squared_error(np.log(splits["y_test"]), val_preds_log, squared=False)
print("RMSE:", rmse)

# --- Predict on test set ---
test_preds_log = learner.predict({"_skrub_X": test_data})

# --- Save test predictions to file ---
output = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": np.exp(test_preds_log)
})
output.to_csv("./working/submission.csv", index=False)