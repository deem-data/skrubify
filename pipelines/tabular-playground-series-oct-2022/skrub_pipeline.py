import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
import skrub

from time import time

t0 = time()

# Read dtypes and replace 'float16' with 'float32'
dtypes_df = pd.read_csv("./input/train_dtypes.csv")
dtypes = {
    k: (v if v != "float16" else "float32")
    for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)
}
dtypes = skrub.var("schema", dtypes)
t1 = time()
print("Loaded data schema and start Skrub ML pipeline definition")

# Read and concatenate training data
file_names = np.ndarray([f"./input/train_{i}.csv" for i in range(10)], dtype=str,)
file_names_skrub = skrub.var("file_names", file_names).skb.subsample(n=1)


def read_and_concat_frame(names, schema):
    frames = [pd.read_csv(f_name, dtype=schema) for f_name in names]
    single_frame = pd.concat(frames, ignore_index=True)
    return single_frame


train_df = file_names_skrub.skb.apply_func(read_and_concat_frame, schema=dtypes).skb.subsample(n=100)

# Prepare the data
X = train_df.drop(
    [
        "game_num",
        "event_id",
        "event_time",
        "player_scoring_next",
        "team_scoring_next",
        "team_A_scoring_within_10sec",
        "team_B_scoring_within_10sec",
    ],
    axis=1,
).skb.mark_as_X()
y = train_df[["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]].skb.mark_as_y()

# Train the model for team A
model_A = lgb.LGBMClassifier(force_col_wise=True)
preds_A = X.skb.apply(model_A, y=y.iloc[:, 0])
# Train the model for team B
model_B = lgb.LGBMClassifier(force_col_wise=True)
preds_B = X.skb.apply(model_B, y=y.iloc[:, 1])

def merge_predictions(a, b):
    if isinstance(a,pd.DataFrame):
        return pd.concat([a, b], axis=1)
    else:
        return [a,b]

preds = preds_A.skb.apply_func(merge_predictions, preds_B)

learner = preds.skb.make_learner()
t3 = time()
print("Defined Skrub pipeline")

# Split the data
splits = preds.skb.train_test_split(test_size=0.2, random_state=42)

print("Using Skrub learner for training and prediction")

learner.fit(splits["train"])
val_preds = learner.predict_proba(splits["test"])
val_preds = pd.DataFrame(
    {
        "team_A_scoring_within_10sec": val_preds[0][:,1],
        "team_B_scoring_within_10sec": val_preds[1][:,1],
    }
)
# Calculate log loss
val_log_loss = log_loss(splits["y_test"], val_preds)  # val_log_loss = log_loss(split["y_test"], val_preds)
print(f"Validation Log Loss: {val_log_loss}")
t4 = time()

# Predict on test set
test_dtypes_df = pd.read_csv("./input/test_dtypes.csv")
test_dtypes = {
    k: (v if v != "float16" else "float32")
    for (k, v) in zip(test_dtypes_df.column, test_dtypes_df.dtype)
}
test_preds = learner.predict_proba({"file_names":["./input/test.csv"], "schema": test_dtypes})
test_preds = pd.DataFrame(
    {
        "team_A_scoring_within_10sec": test_preds[0][:,1],
        "team_B_scoring_within_10sec": test_preds[1][:,1],
    }
)
test_preds.to_csv("./working/submission_skrub.csv", index=False)
t5 = time()
print("Schema      : ", t1 - t0)
print("Pipeline Def: ", t3 - t1)
print("Fit & score : ", t4 - t3)
print("Test        : ", t5 - t4)
print("Total time  : ", t5 - t0)
