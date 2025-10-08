import skrub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load the data and create a skrub variable for the plan
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Prepare features and labels, mark them for skrub
X = data["comment_text"].skb.mark_as_X()
y = data.iloc[:, 2:].skb.mark_as_y()

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_tfidf = X.skb.apply(tfidf_vectorizer)

# For multilabel, fit a separate model for each label and concatenate predictions
models = []
preds = []
for label in y.columns:
    lr = LogisticRegression(C=1.0, solver="liblinear")
    pred = X_tfidf.skb.apply(lr, y=y[label])
    preds.append(pred)
pred_all = preds[0].skb.concat(preds[1:])

# Split for validation
splits = pred_all.skb.train_test_split(test_size=0.2, random_state=42)
learner = pred_all.skb.make_learner()
learner.fit(splits["train"])

# Predict probabilities for each label
y_pred = learner.predict_proba(splits["test"])

# Compute ROC AUC for each label
scores = []
for i, label in enumerate(y.columns):
    score = roc_auc_score(splits["y_test"][label], y_pred[:, i*2+1])
    scores.append(score)
    print(f"ROC AUC for {label}: {score}")

mean_auc = sum(scores) / len(scores)
print(f"Mean column-wise ROC AUC: {mean_auc}")