import skrub
import pandas as pd
from sklearn.dummy import DummyClassifier

def printer(x):
    print(x)
    return x

df = pd.DataFrame({"Names" : ["Mike", "Nathalie"], "label" : [0, 1]})
df2 = pd.DataFrame({"Names" : ["Nat", "Mikey"], "label" : [0, 1]})
df3 = pd.DataFrame({"Names" : ["1", "22"], "label" : [0, 1]})

df = skrub.var("input", df)
# some dummy transformation
df = df.assign(Names=df["Names"].apply(lambda x: x + " Test"))
df = df.skb.apply_func(printer)

X = df.drop("label", axis=1).skb.mark_as_X()
y = df[["label"]].skb.mark_as_y()

preprocess_X = X.assign(Names=X["Names"].apply(len))
preprocess_X = preprocess_X.skb.apply_func(printer)

classifier = DummyClassifier()
pred = preprocess_X.skb.apply(classifier, y=y)

print("----------------------------------")
pred.skb.eval({"input" : df2})

learner = pred.skb.make_learner()

print("----------------------------------")
# with _skrub_X and _skrub_y we can pass values directly to our marked X,y
# skipping the dummy transformation
env = {"_skrub_X" : df3[["Names"]], "_skrub_y" : df3[["label"]]}
learner.fit(env)


print("----------------------------------\nsplitting:")
splits = pred.skb.train_test_split(test_size=0.5)
print("----------------------------------")
learner.fit(splits["train"])
