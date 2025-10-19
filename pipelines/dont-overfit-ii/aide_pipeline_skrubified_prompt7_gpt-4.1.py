import skrub
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Load data and create skrub variable
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Mark y and X
y = data["target"].skb.mark_as_y()
X = data.drop(["id", "target"], axis=1).skb.mark_as_X()

# Define model
model = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
pred = X.skb.apply(model, y=y)

# Prepare cross-validation splits
splits = pred.skb.cv_split(StratifiedKFold(n_splits=10, shuffle=True, random_state=42))

# Create learner
learner = pred.skb.make_learner()

auc_scores = []
for fold in splits:
    learner.fit(fold["train"])
    y_pred_prob = learner.predict_proba(fold["test"])[:, 1]
    auc = roc_auc_score(fold["y_test"], y_pred_prob)
    auc_scores.append(auc)

average_auc_score = sum(auc_scores) / len(auc_scores)
print(f"Average AUC-ROC score: {average_auc_score}")

# Train on full data
learner.fit({"_skrub_X": data.drop(["id", "target"], axis=1), "_skrub_y": data["target"]})

# Predict on test
test_data = pd.read_csv("./input/test.csv")
y_pred_test = learner.predict_proba({"_skrub_X": test_data.drop("id", axis=1)})[:, 1]
test_data["target"] = y_pred_test

# Save submission
submission_file = "./working/submission.csv"
test_data[["id", "target"]].to_csv(submission_file, index=False)