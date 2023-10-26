import pyarrow.parquet as pq
import json
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import tensorflow as tf


stop_words0 = list(stopwords.words("english"))

for i in string.punctuation:
    stop_words0.append(i)

stop_words0.append("<unk>")
stop_words0.append("unk")
stop_words0.append("n't")

print(stop_words0)


"""
data = pq.read_table("md_gender_bias-train-convai.parquet")

lista = []

for i in range(len(data)):
    lista.append((str(data[0][i]), data[1][i].as_py()))

data = pq.read_table("md_gender_bias-train-subs.parquet")

for i in range(len(data)):
    lista.append((str(data[0][i]), data[1][i].as_py()))

with open("subsandconv.json", "w") as fp:
    json.dump(lista, fp)

"""

"""
for i in range(len(data)):
    temp = []
    labels.append(data[i][1])
    sentence = word_tokenize(data[i][0])
    for word in sentence:
        if word.casefold() not in stop_words:
            temp.append(word)
    filtered_list.append(temp)

with open("text.json", "w") as fp:
    json.dump(filtered_list, fp)

with open("labels.json", "w") as fp:
    json.dump(labels, fp)
print(filtered_list)
print(labels)

"""

with open("subsandconv.json", "r") as fp:
    data = json.load(fp)

random.shuffle(data)

text = []
labels = []

for i in range(len(data)):
    text.append(data[i][0])
    labels.append(data[i][1])
with tf.device('/GPU:0'):
    vectorizer = CountVectorizer(stop_words=stop_words0, ngram_range=(1, 3), min_df=3, analyzer='word')
    dtm = vectorizer.fit_transform(text)
    naive_bayes_classifier = MultinomialNB().fit(dtm, labels)

    X_test_tfidf = vectorizer.transform(text)
    y_pred = naive_bayes_classifier.predict(X_test_tfidf)
    precision = metrics.precision_score(labels, y_pred)

    print(precision)

    pickle.dump(naive_bayes_classifier, open("classifierSC.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizerSC.pkl", "wb"))


