import skrub
import pandas as pd
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
drop_cols = [
    "game_num",
    "event_id",
    "event_time",
    "player_scoring_next",
    "team_scoring_next",
    "team_A_scoring_within_10sec",
    "team_B_scoring_within_10sec",
]
X = data.drop(drop_cols, axis=1).skb.mark_as_X()
y_A = data["team_A_scoring_within_10sec"].skb.mark_as_y("A")
y_B = data["team_B_scoring_within_10sec"].skb.mark_as_y("B")

# Model for team A
model_A = lgb.LGBMClassifier()
pred_A = X.skb.apply(model_A, y=y_A)

# Model for team B
model_B = lgb.LGBMClassifier()
pred_B = X.skb.apply(model_B, y=y_B)

# Combine predictions into a DataFrame
preds = skrub.concat(
    [
        pred_A.rename("team_A_scoring_within_10sec"),
        pred_B.rename("team_B_scoring_within_10sec"),
    ],
    axis=1,
)

# Split the data
splits = preds.skb.train_test_split(test_size=0.2, random_state=42)

# Create learner
learner = preds.skb.make_learner()

# Fit the pipeline
learner.fit(splits["train"])

# Predict on validation set
val_preds = learner.predict(splits["test"])
y_val = pd.DataFrame({
    "team_A_scoring_within_10sec": splits["y_A_test"],
    "team_B_scoring_within_10sec": splits["y_B_test"],
})
val_log_loss = log_loss(y_val, val_preds)
print(f"Validation Log Loss: {val_log_loss}")

# Predict on test set
test_dtypes_df = pd.read_csv("./input/test_dtypes.csv")
test_dtypes = {
    k: (v if v != "float16" else "float32")
    for (k, v) in zip(test_dtypes_df.column, test_dtypes_df.dtype)
}
test_df = pd.read_csv("./input/test.csv", dtype=test_dtypes)
X_test = test_df.drop(["id"], axis=1)

test_preds = learner.predict({"_skrub_X": X_test})

# Prepare submission
submission = pd.concat([test_df["id"], test_preds], axis=1)
submission.to_csv("./working/submission_skrub.csv", index=False)
t1 = time()
print("Total time: ", t1 - t0)