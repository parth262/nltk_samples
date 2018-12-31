import pickle
import random
from statistics import mode
import nltk
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC


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


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = [w.lower() for w in movie_reviews.words()]

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


features_set = [(find_features(rev), category) for (rev, category) in documents]

training_set = features_set[:1900]
testing_set = features_set[1900:]

# clf = nltk.NaiveBayesClassifier.train(training_set)

clf_file = open("resources/nb_basic_classifier.pickle", "rb")
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

print("Classification:", vote_clf.classify(testing_set[0][0]), "Confidence %:", vote_clf.confidence(testing_set[0][0])*100)
print("Classification:", vote_clf.classify(testing_set[1][0]), "Confidence %:", vote_clf.confidence(testing_set[1][0])*100)
print("Classification:", vote_clf.classify(testing_set[2][0]), "Confidence %:", vote_clf.confidence(testing_set[2][0])*100)
print("Classification:", vote_clf.classify(testing_set[3][0]), "Confidence %:", vote_clf.confidence(testing_set[3][0])*100)
print("Classification:", vote_clf.classify(testing_set[4][0]), "Confidence %:", vote_clf.confidence(testing_set[4][0])*100)
print("Classification:", vote_clf.classify(testing_set[5][0]), "Confidence %:", vote_clf.confidence(testing_set[5][0])*100)
