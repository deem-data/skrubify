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
train_data = skrub.var("train_data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)
test_data = pd.read_csv("./input/test.csv")

# --- Separate target and predictors ---
y = train_data["SalePrice"].skb.mark_as_y()
X = train_data.drop(["SalePrice"], axis=1).skb.mark_as_X()

# --- Preprocessing transformers ---
numerical_cols = X.select_dtypes(exclude=["object"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

numerical_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# --- Preprocessing step ---
X_preprocessed = X.skb.apply(preprocessor)

# --- Log-transform target ---
def log_transform(y):
    return np.log(y)
y_log = y.skb.apply_func(log_transform)

# --- Model ---
model = GradientBoostingRegressor()
pred = X_preprocessed.skb.apply(model, y=y_log)

# --- Train/val split ---
splits = pred.skb.train_test_split(test_size=0.2, random_state=0)

# --- Create learner ---
learner = pred.skb.make_learner()

# --- Train ---
learner.fit(splits["train"])

# --- Predict on validation ---
y_pred_val = learner.predict(splits["test"])

# --- Evaluate ---
score = mean_squared_error(np.log(splits["y_test"]), y_pred_val, squared=False)
print("RMSE:", score)

# --- Predict on test set ---
test_preds = learner.predict({"_skrub_X": test_data})

# --- Save submission ---
output = pd.DataFrame({"Id": test_data.Id, "SalePrice": np.exp(test_preds)})
output.to_csv("./working/submission.csv", index=False)