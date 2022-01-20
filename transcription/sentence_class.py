import pandas as pd

import re

from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("data.csv", encoding="ISO-8859-1")

X = df['Sentence']

translations = {'Chief Complaint': 1, 'Medical History': 2, 'Current Condition': 3, 'Social History': 4, 'Family History': 5, 'Medical Action': 6, 'Client Details': 8, 'Other': 7}

documents = []

for sen in range(0, len(X)):

    # Replace all the numbers
    document = re.sub(r'\d', '$', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Converting to Lowercase
    document = document.lower()

    documents.append(document)

print(documents)

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words='english')
X = vectorizer.fit_transform(documents).toarray()

print(X)

y = df['Tag']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

values = [50, 100, 200, 500, 1000]

for v in values:
    model = RandomForestClassifier(n_estimators=v)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)
