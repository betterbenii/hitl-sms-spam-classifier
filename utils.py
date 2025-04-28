# utils.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath, encoding='latin-1')
    
    # Sometimes the Kaggle version has extra unnamed columns
    df = df[['v1', 'v2']]  # v1 = label, v2 = message
    df.columns = ['label', 'text']
    
    # Encode labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df

def prepare_train_test(df, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=test_size, random_state=random_state, stratify=df['label']
    )
    return X_train, X_test, y_train, y_test

def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return vectorizer, X_train_vec, X_test_vec
