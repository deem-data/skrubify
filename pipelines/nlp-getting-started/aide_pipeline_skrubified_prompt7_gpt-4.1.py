import skrub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Define the DataOps plan
data = skrub.var("data", train_data).skb.subsample(n=100)

X = data["text"].skb.mark_as_X()
y = data["target"].skb.mark_as_y()

vectorizer = TfidfVectorizer()
X_vec = X.skb.apply(vectorizer)

model = LogisticRegression()
pred = X_vec.skb.apply(model, y=y)

splits = pred.skb.train_test_split(test_size=0.2, random_state=42)
learner = pred.skb.make_learner()
learner.fit(splits["train"])

val_predictions = learner.predict(splits["test"])
f1 = f1_score(splits["y_test"], val_predictions)
print(f"F1 Score on the validation set: {f1}")

test_predictions = learner.predict({"_skrub_X": test_data["text"]})
submission = pd.DataFrame({"id": test_data["id"], "target": test_predictions})
submission.to_csv("./working/submission_skrub.csv", index=False)