import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import skrub
from skrub import selectors as s

# Load the training data and start the DataOps plan
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Separate target from predictors
y = data["SalePrice"].skb.mark_as_y()
X = data.drop(["SalePrice"], axis=1).skb.mark_as_X()

# Apply log transformation to the target variable
y_log = y.skb.apply_func(np.log)

# --- Define Preprocessing Steps in the Skrub Plan ---

# Define selectors for numerical and categorical columns
numerical_selector = s.filter(lambda col: col.dtype != "object")
categorical_selector = s.filter(lambda col: col.dtype == "object")

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy="median")
X_num = X.skb.select(numerical_selector)
X_num_imputed = X_num.skb.apply(numerical_transformer)

# Preprocessing for categorical data
# Step 1: Impute missing values
categorical_imputer = SimpleImputer(strategy="most_frequent")
X_cat = X.skb.select(categorical_selector)
X_cat_imputed = X_cat.skb.apply(categorical_imputer)

# Step 2: One-hot encode
# Skrub requires dense output from transformers
onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat_encoded = X_cat_imputed.skb.apply(onehot_encoder)

# Bundle preprocessed numerical and categorical data
X_processed = X_num_imputed.skb.concat([X_cat_encoded], axis=1)

# Define the model
model = GradientBoostingRegressor(random_state=42)

# Apply the model to the processed features and the log-transformed target
preds_log = X_processed.skb.apply(model, y=y_log)

# Apply inverse transformation to the predictions
preds = preds_log.skb.apply_func(lambda x: x if isinstance(x, GradientBoostingRegressor) else np.exp(x))
# The DataOps plan is now complete

# Split data into training and validation subsets using the plan
splits = preds.skb.train_test_split(
    train_size=0.8, test_size=0.2, random_state=0
)

# Create a trainable learner object from the entire pipeline
learner = preds.skb.make_learner()

# Fit the learner on the training data
# Skrub automatically handles all preprocessing and model training
learner.fit(splits["train"])

# Get predictions on the validation set
y_pred_valid = learner.predict(splits["test"])

# Evaluate the model
# The target `splits['y_test']` is the original, non-transformed SalePrice
score = np.sqrt(mean_squared_error(splits["y_test"], y_pred_valid))
print("RMSE:", score)

# Load test data
test_data = pd.read_csv("./input/test.csv")

# Get predictions for the test data
# The learner applies the entire preprocessing pipeline automatically
test_preds = learner.predict({"_skrub_X": test_data})

# Save test predictions to file
# The predictions are already inverse-transformed by the pipeline
output = pd.DataFrame({"Id": test_data.Id, "SalePrice": test_preds})
output.to_csv("./working/submission_skrub_gemini.csv", index=False)