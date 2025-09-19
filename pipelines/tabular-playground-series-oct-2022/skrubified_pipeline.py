import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from time import time
import skrub

t0 = time()

# --- Read dtypes and replace 'float16' with 'float32' ---
dtypes_df = pd.read_csv("./input/train_dtypes.csv")
dtypes = {
    k: (v if v != "float16" else "float32")
    for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)
}

# --- Read and concatenate training data ---
train_dfs = [pd.read_csv(f"./input/train_{i}.csv", dtype=dtypes) for i in range(10)]
train_df = pd.concat(train_dfs, ignore_index=True)
train_df = skrub.var("train_df", train_df)

# --- Prepare the data ---
drop_cols = [
    "game_num",
    "event_id",
    "event_time",
    "player_scoring_next",
    "team_scoring_next",
    "team_A_scoring_within_10sec",
    "team_B_scoring_within_10sec",
]
X = train_df.drop(drop_cols, axis=1).skb.mark_as_X()
y = train_df[["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]].skb.mark_as_y()

# --- Model for team A ---
model_A = lgb.LGBMClassifier()
pred_A = X.skb.apply(model_A, y=y.iloc[:, 0])

# --- Model for team B ---
model_B = lgb.LGBMClassifier()
pred_B = X.skb.apply(model_B, y=y.iloc[:, 1])

# --- Train/val split ---
split_A = pred_A.skb.train_test_split(test_size=0.2, random_state=42)
split_B = pred_B.skb.train_test_split(test_size=0.2, random_state=42)

# --- Learners ---
learner_A = pred_A.skb.make_learner()
learner_B = pred_B.skb.make_learner()

# --- Train ---
learner_A.fit(split_A["train"])
learner_B.fit(split_B["train"])

# --- Predict on validation set ---
val_preds_A = learner_A.predict_proba(split_A["test"])[:, 1]
val_preds_B = learner_B.predict_proba(split_B["test"])[:, 1]
val_preds = pd.DataFrame(
    {
        "team_A_scoring_within_10sec": val_preds_A,
        "team_B_scoring_within_10sec": val_preds_B,
    }
)
y_val = split_A["y_test"].reset_index(drop=True)
val_log_loss = log_loss(y_val, val_preds)
print(f"Validation Log Loss: {val_log_loss}")

# --- Predict on test set ---
test_dtypes_df = pd.read_csv("./input/test_dtypes.csv")
test_dtypes = {
    k: (v if v != "float16" else "float32")
    for (k, v) in zip(test_dtypes_df.column, test_dtypes_df.dtype)
}
test_df = pd.read_csv("./input/test.csv", dtype=test_dtypes)
X_test = test_df.drop(["id"], axis=1)

test_preds_A = learner_A.predict_proba({"_skrub_X": X_test})[:, 1]
test_preds_B = learner_B.predict_proba({"_skrub_X": X_test})[:, 1]
test_preds = pd.DataFrame(
    {
        "team_A_scoring_within_10sec": test_preds_A,
        "team_B_scoring_within_10sec": test_preds_B,
    }
)

# --- Prepare submission ---
submission = pd.concat([test_df["id"], test_preds], axis=1)
submission.to_csv("./working/submission.csv", index=False)

t1 = time()
print("Total time: ", t1 - t0)