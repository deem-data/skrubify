import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.metrics import log_loss
from time import time

t0 = time()

# Read dtypes and replace 'float16' with 'float32'
dtypes_df = pd.read_csv("./input/train_dtypes.csv")
dtypes = {
    k: (v if v != "float16" else "float32")
    for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)
}

# Read and concatenate training data
train_dfs = [pd.read_csv(f"./input/train_{i}.csv", dtype=dtypes) for i in range(10)]
train_df = pd.concat(train_dfs, ignore_index=True)

# Start Skrub DataOps plan
data = skrub.var("data", train_df).skb.subsample(n=100)

# Prepare the data
X = data.drop(
    [
        "game_num",
        "event_id",
        "event_time",
        "player_scoring_next",
        "team_scoring_next",
        "team_A_scoring_within_10sec",
        "team_B_scoring_within_10sec",
    ],
    axis=1
).skb.mark_as_X()
y = data[["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]].skb.mark_as_y()

# Model for team A
model_A = lgb.LGBMClassifier()
pred_A = X.skb.apply(model_A, y=y["team_A_scoring_within_10sec"])

# Model for team B
model_B = lgb.LGBMClassifier()
pred_B = X.skb.apply(model_B, y=y["team_B_scoring_within_10sec"])

@skrub.deferred
def combine_preds(pred_A, pred_B):
    import numpy as np
    import pandas as pd
    # pred_A and pred_B are np.ndarray (predict_proba) or pd.Series (predict)
    if isinstance(pred_A, np.ndarray):
        return pd.DataFrame({
            "team_A_scoring_within_10sec": pred_A[:, 1],
            "team_B_scoring_within_10sec": pred_B[:, 1],
        })
    elif isinstance(pred_A, pd.Series) or isinstance(pred_A, pd.DataFrame):
        return pd.DataFrame({
            "team_A_scoring_within_10sec": pred_A,
            "team_B_scoring_within_10sec": pred_B,
        })
    else:
        return (pred_A, pred_B)

pred = combine_preds(pred_A, pred_B)

# Split for validation
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Validation
val_preds = learner.predict_proba(splits["test"])
val_log_loss = log_loss(splits["y_test"], val_preds)
print(f"Validation Log Loss: {val_log_loss}")

# Test set
test_dtypes_df = pd.read_csv("./input/test_dtypes.csv")
test_dtypes = {
    k: (v if v != "float16" else "float32")
    for (k, v) in zip(test_dtypes_df.column, test_dtypes_df.dtype)
}
test_df = pd.read_csv("./input/test.csv", dtype=test_dtypes)
X_test = test_df.drop(["id"], axis=1)

test_preds = learner.predict_proba({"_skrub_X": X_test})

submission = pd.concat([test_df["id"], test_preds], axis=1)
submission.to_csv("./working/submission_skrub.csv", index=False)

t1 = time()
print("Total time: ", t1 - t0)