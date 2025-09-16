import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from concurrent.futures import ProcessPoolExecutor
import skrub

from time import time

t0 = time()

# Read dtypes and replace 'float16' with 'float32'
dtypes_df = pd.read_csv("./input/train_dtypes.csv")
dtypes = {
    k: (v if v != "float16" else "float32")
    for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)
}
t1 = time()
print("Loaded data schema and starting to read in data in parallel")

# Read and concatenate training data

def read_csv_file(i):
    return pd.read_csv(f"./input/train_{i}.csv", dtype=dtypes)

# parallel read
with ProcessPoolExecutor() as executor:
    train_dfs = list(executor.map(read_csv_file, range(10)))

# concatenate
train_df = pd.concat(train_dfs, ignore_index=True)
t2 = time()
print("Read and merge train data")

# manually clean up memory
del train_dfs

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
)
y = train_df[["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]]

# Skrub code starts here:
X_skrub = skrub.X() # init var with empty to avoid preview computation
y_skrub = skrub.y()

# Train the model for team A
model_A = lgb.LGBMClassifier(force_col_wise=True)
preds_A = X_skrub.skb.apply(model_A, y=y_skrub.iloc[:, 0])
# Train the model for team B
model_B = lgb.LGBMClassifier(force_col_wise=True)
preds_B = X_skrub.skb.apply(model_B, y=y_skrub.iloc[:, 1])

learner_A = preds_A.skb.make_learner()
learner_B = preds_B.skb.make_learner()
t3 = time()
print("Defined Skrub pipeline")

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
del X,y
train_env_dict = {"X" : X_train, "y" : y_train}
val_env_dict = {"X" : X_val, "y" : y_val}

print("Using Skrub learner for training and prediction")

learner_A.fit(train_env_dict)
learner_B.fit(train_env_dict)
prob_A = learner_A.predict_proba(val_env_dict)[:, 1]
prob_B = learner_B.predict_proba(val_env_dict)[:, 1]

# split = preds.skb.train_test_split()
# learner_A.fit(split["train"])
# learner_B.fit(split["train"])
# prob_A = learner_A.predict_proba(split["test"])[:, 1]
# prob_B = learner_B.predict_proba(split["test"])[:, 1]

# Combine predictions
val_preds = pd.DataFrame(
    {
        "team_A_scoring_within_10sec": prob_A,
        "team_B_scoring_within_10sec": prob_B,
    }
)

# Calculate log loss
val_log_loss = log_loss(y_val, val_preds) # val_log_loss = log_loss(split["y_test"], val_preds)
print(f"Validation Log Loss: {val_log_loss}")

t4 = time()
print("Schema      : ", t1 - t0)
print("Data        : ", t2 - t1)
print("Pipeline Def: ", t3 - t2)
print("Schema      : ", t4 - t3)
print("Total time  : ", t4 - t0)