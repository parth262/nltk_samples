import nltk
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


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

clf = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo accuracy score:", nltk.classify.accuracy(clf, testing_set))
clf.show_most_informative_features(15)
