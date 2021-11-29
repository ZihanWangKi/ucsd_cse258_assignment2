from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import json
import os.path
import random
import sys
import numpy as np
import nltk

dataset = sys.argv[1]
assert dataset in ["amazon", "goodreads"]

path = f"/data/zihan/coursework/cse258/assignment2/data/{dataset}"

with open(os.path.join(path, "train.json")) as f:
    train_data = [json.loads(x) for x in f.readlines()]

with open(os.path.join(path, "eval.json")) as f:
    eval_data = [json.loads(x) for x in f.readlines()]

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                        encoding='latin-1', ngram_range=(1, 2), stop_words='english')
train_features = tfidf.fit_transform([" ".join(nltk.word_tokenize(d["review_text"])) for d in train_data]).toarray()
train_labels = [d["label"] for d in train_data]

clf = MultinomialNB().fit(train_features, train_labels)

predict_labels = clf.predict(tfidf.transform([" ".join(nltk.word_tokenize(d["review_text"])) for d in eval_data]))
eval_labels = [d["label"] for d in train_data]

acc = (predict_labels == np.array(eval_labels)).sum() / len(eval_labels)
print(dataset, acc)