import numpy as np
import pandas as pd
import pickle
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from utils.preprocessor import TextPreprocessor

nltk.download('stopwords')

column_names = [
    'target',
    'ids',
    'date',
    'flag',
    'user',
    'text'
]

twitter_data = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    names=column_names,
    encoding='ISO-8859-1'
)

twitter_data.replace({'target': {4: 1}}, inplace=True)

X = twitter_data['text'].values
y = twitter_data['target'].values
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=2
)

pipeline = Pipeline([
    (
        'preprocessing',
        TextPreprocessor()
    ),
    (
        'tfidf',
        TfidfVectorizer(
            max_features=50000,
            ngram_range=(1,3),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95
        )
    ),
    (
        'model',
        LinearSVC(
            C=2,
            max_iter=2000
        )
    )
])

print("Training started...")
pipeline.fit(X_train, y_train)
print("Training completed!")

pred = pipeline.predict(X_test)
print(
    "Accuracy:",
    accuracy_score(y_test, pred)
)

pickle.dump(
    pipeline,
    open('sentiment_pipeline.pkl', 'wb')
)

print("Model Saved!")
