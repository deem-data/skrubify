import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# Load train data and create skrub var for pipeline definition
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Mark target as y and features as X
y = data["target"].skb.mark_as_y()
X = data.drop(["row_id", "target"], axis=1).skb.mark_as_X()

# Select numeric columns for imputation
numeric_selector = skrub.selectors.filter(lambda col: np.issubdtype(col.dtype, np.number))
X_numeric = X.skb.select(numeric_selector)
X_numeric_imputed = X_numeric.skb.apply_func(lambda df: df.fillna(df.median()))

# For non-numeric columns, just keep as is
non_numeric_selector = skrub.selectors.filter(lambda col: not np.issubdtype(col.dtype, np.number))
X_non_numeric = X.skb.select(non_numeric_selector)

# Concatenate imputed numeric and untouched non-numeric columns
X_preprocessed = X_numeric_imputed.skb.concat([X_non_numeric], axis=1)

# Model
model = LGBMRegressor()
pred = X_preprocessed.skb.apply(model, y=y)

# Prepare cross-validation splits
splits = pred.skb.kfold(n_splits=10, shuffle=True, random_state=42)

# Create learner
learner = pred.skb.make_learner()

# Cross-validation loop
mae_scores = []
for split in splits:
    learner.fit(split["train"])
    y_pred = learner.predict(split["test"])
    mae = mean_absolute_error(split["y_test"], y_pred)
    mae_scores.append(mae)

print(f"Average MAE: {np.mean(mae_scores)}")

# Predict on test set
test_data = pd.read_csv("./input/test.csv")
test_features = test_data.drop(["row_id"], axis=1)
y_pred_test = learner.predict({"_skrub_X": test_features})

# Save predictions
submission = pd.DataFrame({"row_id": test_data["row_id"], "target": y_pred_test})
submission.to_csv("./working/submission_skrub.csv", index=False)