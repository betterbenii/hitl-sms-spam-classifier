# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
from model import load_model_and_vectorizer
from utils import load_and_preprocess_data
import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = 'secretkey123'
   
global model, vectorizer
# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Load the dataset
data = load_and_preprocess_data("data/spam.csv")

# Feedback storage file
FEEDBACK_FILE = "feedback/feedback.json"
os.makedirs("feedback", exist_ok=True)

# Load existing feedback if any
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "r") as f:
        feedback_data = json.load(f)
else:
    feedback_data = []

# Helper to find uncertain samples
def get_uncertain_sample():
    texts = data['text'].values
    X_vec = vectorizer.transform(texts)
    probs = model.predict_proba(X_vec)
    uncertainty = 1 - np.max(probs, axis=1)
    
    sorted_indices = np.argsort(uncertainty)[::-1]  # most uncertain first
    
    for idx in sorted_indices:
        text = texts[idx]
        label = model.predict(X_vec[idx])[0]
        # Check if already labeled
        if not any(fb['text'] == text for fb in feedback_data):
            return idx, text, label
    return None, None, None

@app.route('/')
def home():
    idx, text, prediction = get_uncertain_sample()
    if text is None:
        return "No more uncertain samples to review! 🎉"
    label_name = "Spam" if prediction == 1 else "Ham"
    return render_template('review.html', idx=idx, text=text, prediction=label_name)

@app.route('/feedback', methods=['POST'])
def feedback():
    idx = int(request.form['idx'])
    action = request.form['action']
    
    text = data.iloc[idx]['text']
    
    if action == 'confirm':
        label = model.predict(vectorizer.transform([text]))[0]
    elif action == 'correct_spam':
        label = 1
    elif action == 'correct_ham':
        label = 0
    else:
        return redirect(url_for('home'))  # Skip

    feedback_entry = {
        'text': text,
        'label': int(label)
    }
    feedback_data.append(feedback_entry)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_data, f, indent=4)
    
    return redirect(url_for('home'))
@app.route('/retrain', methods=['POST'])
def retrain():
    global model, vectorizer

    if not feedback_data:
        flash("No feedback collected yet to retrain.", "error")
        return redirect(url_for('home'))

    # Load original data
    original_data = load_and_preprocess_data("data/spam.csv")

    # Load fixed validation set
    with open("validation_set.json", "r") as f:
        validation_data = json.load(f)

    val_texts = validation_data['texts']
    val_labels = validation_data['labels']

    # --------- 1. Evaluate INITIAL model (before retraining) ---------
    X_val_vec_before = vectorizer.transform(val_texts)
    from sklearn.metrics import accuracy_score
    initial_accuracy = accuracy_score(val_labels, model.predict(X_val_vec_before))

    # --------- 2. Retrain Model ---------
    feedback_df = pd.DataFrame(feedback_data)

    training_texts = pd.concat([
        original_data['text'],
        feedback_df['text']
    ])

    training_labels = pd.concat([
        original_data['label'],
        feedback_df['label']
    ])

    new_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = new_vectorizer.fit_transform(training_texts)

    new_model = LogisticRegression(max_iter=1000)
    new_model.fit(X_train_vec, training_labels)

    # Save model and vectorizer
    joblib.dump(new_model, "model.joblib")
    joblib.dump(new_vectorizer, "vectorizer.joblib")

    # Reload model
    model, vectorizer = load_model_and_vectorizer()

    # --------- 3. Evaluate RETRAINED model (after retraining) ---------
    X_val_vec_after = vectorizer.transform(val_texts)
    retrained_accuracy = accuracy_score(val_labels, model.predict(X_val_vec_after))

    # --------- 4. Flash Message ---------
    flash(f"✅ Model retrained successfully!📊 Initial Accuracy: {initial_accuracy:.4f} 📈 New Accuracy: {retrained_accuracy:.4f}", "success")

    # --------- 5. Save to Retrain Log ---------
    log_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "feedback_count": len(feedback_data),
        "initial_accuracy": round(initial_accuracy, 4),
        "retrained_accuracy": round(retrained_accuracy, 4)
    }

    log_file = "retrain_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_data = json.load(f)
    else:
        log_data = []

    log_data.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)

    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
