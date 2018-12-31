import pickle
import random
from statistics import mode
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize


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


save_documents = open("resources/documents.pickle", "rb")
documents = pickle.load(save_documents)
save_documents.close()

save_word_features = open("resources/word_features.pickle", "rb")
word_features = pickle.load(save_word_features)
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

clf_file = open("resources/nb_basic_classifier2.pickle", "rb")
clf = pickle.load(clf_file)
clf_file.close()

clf_file = open("resources/MNB_clf.pickle", "rb")
MNB_clf = pickle.load(clf_file)
clf_file.close()

clf_file = open("resources/BernoulliNB_clf.pickle", "rb")
BernoulliNB_clf = pickle.load(clf_file)
clf_file.close()

clf_file = open("resources/LogisticRegression_clf.pickle", "rb")
LogisticRegression_clf = pickle.load(clf_file)
clf_file.close()

clf_file = open("resources/LinearSVC_clf.pickle", "rb")
LinearSVC_clf = pickle.load(clf_file)
clf_file.close()

vote_clf = VoteClassifier(clf,
                          MNB_clf,
                          BernoulliNB_clf,
                          LogisticRegression_clf,
                          LinearSVC_clf)
# print("vote_clf Algo accuracy score:", nltk.classify.accuracy(vote_clf, testing_set))


def sentiment(text):
    feats = find_features(text)

    return vote_clf.classify(feats), vote_clf.confidence(feats)
