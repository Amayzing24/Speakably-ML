import pandas as pd

import re

from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("medical_action.csv", encoding="ISO-8859-1")

X = df['Sentence']

documents = []

for sen in range(0, len(X)):

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(X[sen]))

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Converting to Lowercase
    document = document.lower()

    documents.append(document)

vectorizer = CountVectorizer(max_features=1500, min_df=1, max_df=0.8, stop_words='english')
X = vectorizer.fit_transform(documents).toarray()

y = df['Tag']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

models = [RandomForestClassifier(), LogisticRegression(), LinearSVC()]

for model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)
