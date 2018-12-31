from nltk.stem import WordNetLemmatizer

le = WordNetLemmatizer()

print(le.lemmatize("cats"))
print(le.lemmatize("cacti"))
print(le.lemmatize("geese"))
print(le.lemmatize("rocks"))
print(le.lemmatize("python"))
print(le.lemmatize("java"))
print(le.lemmatize("better", pos='a'))
 