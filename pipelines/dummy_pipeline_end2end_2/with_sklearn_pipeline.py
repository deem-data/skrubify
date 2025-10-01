import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Load data ---
data = pd.read_csv("./input/train.csv")

# Separate labels
y = data["label"]
X = data.drop("label", axis=1)

# --- Feature engineering function ---
def add_new_feat(df):
    df = df.copy()
    df["new_feat"] = df["feat1"] * df["feat2"]
    return df

# Wrap it as a transformer
feat_engineering = FunctionTransformer(add_new_feat)

# --- Preprocessing ---
# Drop columns ["id", "feat1", "feat3"], then scale remaining
drop_cols = ["id", "feat1", "feat3"]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), [col for col in X.columns if col not in drop_cols])
    ]
)

# --- Full pipeline ---
pipeline = Pipeline(steps=[
    ("feature_engineering", feat_engineering),
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- Train/val split ---
X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=42)

# --- Train ---
pipeline.fit(X_train, y_train)

# --- Evaluate ---
y_pred = pipeline.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")

# --- Predict on test ---
test_data = pd.read_csv("./input/test.csv")
y_pred_test = pipeline.predict(test_data)

# --- Save submission ---
submission = pd.DataFrame({"id": test_data["id"], "label": y_pred_test})
submission.to_csv("./working/submission_pipe.csv", index=False)
