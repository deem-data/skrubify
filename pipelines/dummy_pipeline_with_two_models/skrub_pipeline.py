import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import numpy as np
import skrub

# Load training data
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Separate features and labels
features = data.drop(["label1","label2"], axis=1).skb.mark_as_X()
labels = data[["label1","label2"]].skb.mark_as_y()

# Add features normalization
scaler = StandardScaler()
features_scaled = features.skb.apply(scaler)

# Add models
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
pred1 = features_scaled.skb.apply(model1, y=labels["label1"])

model2 = RandomForestClassifier(n_estimators=100, random_state=42)
pred2 = features_scaled.skb.apply(model2, y=labels["label2"])

# Merge to skrub variables to single
pred = pred1.skb.apply_func(lambda a,b: np.column_stack((a,b)), b=pred2)

# Split data
splits = pred.skb.train_test_split(test_size=0.2, random_state=42)

# Train pipeline
learner = pred.skb.make_learner()
learner.fit(splits["train"])

# Evaluate
y_pred = learner.predict_proba(splits["test"]) # shape (n,2+2)
loss = log_loss(splits["y_test"], y_pred[:,[1,3]]) #
print(f"Validation Accuracy: {loss:.4f}")

# Prepare test data
test_data = pd.read_csv("./input/test.csv")
y_pred_test = learner.predict_proba({"_skrub_X" : test_data})

# Predict and save submission
submission = pd.DataFrame({"id": test_data["id"], "label1_prob": y_pred_test[:,1], "label2_prob": y_pred_test[:,3]})
submission.to_csv("./working/submission_skrub.csv", index=False)