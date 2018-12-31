import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import natural_language_processing.example12b as s

df = pd.read_csv('resources/wine-reviews/winemag-data-130k.csv', nrows=1000)

drop_cols = ['Unnamed: 0', 'taster_twitter_handle', 'variety', 'designation', 'title', 'winery',
             'country', 'province', 'region_1', 'region_2']

df.drop(drop_cols, 1, inplace=True)

df.dropna(inplace=True)

sentiments = []

print(df.shape)

print("starting sentiment calculations.")

for row in df['description']:
    sentiment, confidence = s.sentiment(row)
    if sentiment == 'pos':
        sentiments.append(confidence)
    else:
        sentiments.append(-confidence)

df['sentiments'] = sentiments

print("sentiments calculated")

df.drop(["description"], 1, inplace=True)

print(df.head())

cat_cols = df.select_dtypes(exclude=[np.number]).columns

le = LabelEncoder()

print("starting label encoding.")

for col in df.columns:
    try:
        le.fit(df[col])
        df[col] = le.transform(df[col])
    except TypeError:
        print("error in column: ", col)

print("label encoding done.")

X = df.drop(['price'], 1)
scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LinearRegression()

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print(score)
