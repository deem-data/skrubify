import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss
from time import time
import skrub

t0 = time()

# Read dtypes and replace 'float16' with 'float32'
dtypes_df = pd.read_csv("./input/train_dtypes.csv")
dtypes = {k: (v if v != "float16" else "float32") for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}

# Read and concatenate training data
train_dfs = [pd.read_csv(f"./input/train_{i}.csv", dtype=dtypes) for i in range(10)]
train_df = pd.concat(train_dfs, ignore_index=True)

# Define columns
label_cols = ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]
drop_cols = [
    "game_num",
    "event_id",
    "event_time",
    "player_scoring_next",
    "team_scoring_next",
    "team_A_scoring_within_10sec",
    "team_B_scoring_within_10sec",
]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

# Build Skrub DataOps plan
data = skrub.var("train_df", train_df).skb.subsample(n=100)

y = data[label_cols].skb.mark_as_y()
X_raw = data.skb.mark_as_X()
X = X_raw[feature_cols]

model_A = lgb.LGBMClassifier()
predA = X.skb.apply(model_A, y=y["team_A_scoring_within_10sec"])

model_B = lgb.LGBMClassifier()
predB = X.skb.apply(model_B, y=y["team_B_scoring_within_10sec"])

@skrub.deferred
def concat_probs(predA, predB):
    if isinstance(predA, pd.DataFrame) or isinstance(predA, pd.Series):
        return pd.concat([predA, predB], axis=1)
    elif isinstance(predA, np.ndarray):
        return pd.DataFrame(
            {
                "team_A_scoring_within_10sec": predA[:, 1],
                "team_B_scoring_within_10sec": predB[:, 1],
            }
        )
    else:
        return (predA, predB)

pred = concat_probs(predA, predB)

# Split, train, evaluate
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)
learner = pred.skb.make_learner()
learner.fit(splits["train"])

val_preds = learner.predict_proba(splits["test"])
val_log_loss = log_loss(splits["y_test"], val_preds)
print(f"Validation Log Loss: {val_log_loss}")

# Predict on test set
test_dtypes_df = pd.read_csv("./input/test_dtypes.csv")
test_dtypes = {k: (v if v != "float16" else "float32") for (k, v) in zip(test_dtypes_df.column, test_dtypes_df.dtype)}
test_df = pd.read_csv("./input/test.csv", dtype=test_dtypes)

test_preds = learner.predict_proba({"_skrub_X": test_df})
submission = pd.concat([test_df["id"], test_preds], axis=1)
submission.to_csv("./working/submission.csv", index=False)

t1 = time()
print("Total time: ", t1 - t0)