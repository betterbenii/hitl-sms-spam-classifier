# model.py

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import load_and_preprocess_data, prepare_train_test, vectorize_text

def train_and_save_model(data_path, model_path="model.joblib", vectorizer_path="vectorizer.joblib"):
    # Step 1: Load and preprocess
    df = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    
    # Step 2: Vectorize
    vectorizer, X_train_vec, X_test_vec = vectorize_text(X_train, X_test)
    
    # Step 3: Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Step 4: Save model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Step 5: Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Initial Model Accuracy: {acc:.4f}")
    
def load_model_and_vectorizer(model_path="model.joblib", vectorizer_path="vectorizer.joblib"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
