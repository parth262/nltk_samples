import pickle
import random

import nltk
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from statistics import mode
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC


class VoteClassifier(ClassifierI):
    def labels(self):
        pass

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, featureset):
        votes = []
        for c in self._classifiers:
            v = c.classify(featureset)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


short_pos = open('resources/positive.txt', 'r', errors='ignore').read()
short_neg = open('resources/negative.txt', 'r', errors='ignore').read()

documents = []
all_words = []

allowed_word_types = ["J"]

for p in short_pos.split("\n"):
    documents.append((p, 'pos'))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split("\n"):
    documents.append((p, 'neg'))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("resources/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_word_features = open("resources/word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


features_set = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(features_set)

training_set = features_set[:10000]
testing_set = features_set[10000:]

clf = nltk.NaiveBayesClassifier.train(training_set)

clf_file = open("resources/nb_basic_classifier2.pickle", "wb")
pickle.dump(clf, clf_file)
clf_file.close()

print("Naive Bayes Algo accuracy score:", nltk.classify.accuracy(clf, testing_set))
clf.show_most_informative_features(15)

MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB_clf Algo accuracy score:", nltk.classify.accuracy(MNB_clf, testing_set))

clf_file = open("resources/MNB_clf.pickle", "wb")
pickle.dump(MNB_clf, clf_file)
clf_file.close()

BernoulliNB_clf = SklearnClassifier(BernoulliNB())
BernoulliNB_clf.train(training_set)
print("BernoulliNB_clf Algo accuracy score:", nltk.classify.accuracy(BernoulliNB_clf, testing_set))

clf_file = open("resources/BernoulliNB_clf.pickle", "wb")
pickle.dump(BernoulliNB_clf, clf_file)
clf_file.close()

LogisticRegression_clf = SklearnClassifier(LogisticRegression())
LogisticRegression_clf.train(training_set)
print("LogisticRegression_clf Algo accuracy score:", nltk.classify.accuracy(LogisticRegression_clf, testing_set))

clf_file = open("resources/LogisticRegression_clf.pickle", "wb")
pickle.dump(LogisticRegression_clf, clf_file)
clf_file.close()

# SGDClassifier_clf = SklearnClassifier(SGDClassifier())
# SGDClassifier_clf.train(training_set)
# print("SGDClassifier_clf Algo accuracy score:", nltk.classify.accuracy(SGDClassifier_clf, testing_set))

# SVC_clf = SklearnClassifier(SVC())
# SVC_clf.train(training_set)
# print("SVC_clf Algo accuracy score:", nltk.classify.accuracy(SVC_clf, testing_set))

LinearSVC_clf = SklearnClassifier(LinearSVC())
LinearSVC_clf.train(training_set)
print("LinearSVC_clf Algo accuracy score:", nltk.classify.accuracy(LinearSVC_clf, testing_set))

clf_file = open("resources/LinearSVC_clf.pickle", "wb")
pickle.dump(LinearSVC_clf, clf_file)
clf_file.close()

# NuSVC_clf = SklearnClassifier(NuSVC())
# NuSVC_clf.train(training_set)
# print("NuSVC_clf Algo accuracy score:", nltk.classify.accuracy(NuSVC_clf, testing_set))

# clf_file = open("resources/NuSVC_clf.pickle", "wb")
# pickle.dump(NuSVC_clf, clf_file)
# clf_file.close()

vote_clf = VoteClassifier(clf,
                          MNB_clf,
                          BernoulliNB_clf,
                          LogisticRegression_clf,
                          LinearSVC_clf)
print("vote_clf Algo accuracy score:", nltk.classify.accuracy(vote_clf, testing_set))


def sentiment(text):
    feats = find_features(text)

    return vote_clf.classify(feats), vote_clf.confidence(feats)
