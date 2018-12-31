import pickle
import random
from nltk.tokenize import word_tokenize
from statistics import mode
import nltk
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC
from numba import vectorize


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


def test():
    short_pos = open('resources/positive.txt', 'r', errors='ignore').read()
    short_neg = open('resources/negative.txt', 'r', errors='ignore').read()
    documents = []
    for r in short_pos.split("\n"):
        documents.append((r, 'pos'))

    for r in short_neg.split("\n"):
        documents.append((r, 'neg'))

    all_words = []

    short_pos_words = word_tokenize(short_pos)
    short_neg_words = word_tokenize(short_neg)

    for w in short_pos_words:
        all_words.append(w.lower())

    for w in short_neg_words:
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)

    word_features = list(all_words.keys())[:5000]

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

    # clf = nltk.NaiveBayesClassifier.train(training_set)

    clf_file = open("resources/nb_basic_classifier1.pickle", "rb")
    clf = pickle.load(clf_file)
    clf_file.close()

    print("Naive Bayes Algo accuracy score:", nltk.classify.accuracy(clf, testing_set))
    clf.show_most_informative_features(15)

    MNB_clf = SklearnClassifier(MultinomialNB())
    MNB_clf.train(training_set)
    print("MNB_clf Algo accuracy score:", nltk.classify.accuracy(MNB_clf, testing_set))

    BernoulliNB_clf = SklearnClassifier(BernoulliNB())
    BernoulliNB_clf.train(training_set)
    print("BernoulliNB_clf Algo accuracy score:", nltk.classify.accuracy(BernoulliNB_clf, testing_set))

    LogisticRegression_clf = SklearnClassifier(LogisticRegression())
    LogisticRegression_clf.train(training_set)
    print("LogisticRegression_clf Algo accuracy score:", nltk.classify.accuracy(LogisticRegression_clf, testing_set))

    SGDClassifier_clf = SklearnClassifier(SGDClassifier())
    SGDClassifier_clf.train(training_set)
    print("SGDClassifier_clf Algo accuracy score:", nltk.classify.accuracy(SGDClassifier_clf, testing_set))

    # SVC_clf = SklearnClassifier(SVC())
    # SVC_clf.train(training_set)
    # print("SVC_clf Algo accuracy score:", nltk.classify.accuracy(SVC_clf, testing_set))

    LinearSVC_clf = SklearnClassifier(LinearSVC())
    LinearSVC_clf.train(training_set)
    print("LinearSVC_clf Algo accuracy score:", nltk.classify.accuracy(LinearSVC_clf, testing_set))

    NuSVC_clf = SklearnClassifier(NuSVC())
    NuSVC_clf.train(training_set)
    print("NuSVC_clf Algo accuracy score:", nltk.classify.accuracy(NuSVC_clf, testing_set))

    vote_clf = VoteClassifier(clf,
                              MNB_clf,
                              BernoulliNB_clf,
                              LogisticRegression_clf,
                              SGDClassifier_clf,
                              LinearSVC_clf,
                              NuSVC_clf)
    print("vote_clf Algo accuracy score:", nltk.classify.accuracy(vote_clf, testing_set))


test()
