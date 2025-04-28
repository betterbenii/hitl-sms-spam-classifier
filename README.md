# Human-in-the-Loop SMS Spam Classifier

## Overview

This project implements a Human-in-the-Loop (HITL) Active Learning System for SMS spam classification.  
The goal is to build an initial machine learning model and iteratively improve it through human feedback on uncertain predictions.

The system allows:
- An AI model to predict whether an SMS is spam or ham (not spam)
- A human to confirm, correct, or skip AI predictions
- Retraining the model based on the human-labeled data
- Tracking and visualizing how the model improves over time

## System Architecture

1. Data Layer
   - Dataset: SMS Spam Collection dataset (~5,500 labeled messages)
   - Preprocessing: TF-IDF Vectorization
   - Data split: 80% for training pool, 20% fixed validation set

2. Model Layer
   - Initial Model: Logistic Regression (trained on TF-IDF features)
   - Retraining: Original training data + accumulated human feedback
   - Evaluation: Always evaluated on the same fixed validation set to ensure fairness

3. Interface Layer
   - Web Application: Built using Flask
   - User Interface:
     - Shows uncertain SMS samples based on model confidence
     - Allows the user to Confirm, Correct to Spam, Correct to Ham, or Skip
     - A "Retrain Now" button retrains the model when enough feedback has been collected

4. Logging and Visualization
   - Feedbacks are saved into feedback/feedback.json
   - Retrain logs (accuracy before and after retraining) are saved into retrain_log.json
   - Retrain progress can be visualized using plot_retrain_progress.py

## How the System Works

1. The baseline model is trained on a small subset of the original data.
2. During active learning, the system selects samples where the model has the least confidence.
3. The human user reviews these predictions:
   - Confirm correct predictions
   - Correct wrong predictions
   - Skip samples they are unsure about
4. Once enough feedback is collected, the model is retrained using the original training pool plus the human feedbacks.
5. Model evaluation is always done against the same fixed validation set.
6. The system logs feedback counts and accuracy improvements over time.

## Baseline Model vs Human-in-the-Loop Model

| Aspect | Baseline Model | HITL-Enhanced Model |
|:---|:---|
| Data Used | Only original training set | Original training set + Human feedback |
| Learning Behavior | Static after initial training | Dynamic, improves after every retrain |
| Accuracy | Plateaus after first training | Continues to improve with human corrections |
| Handling Confusion | No correction mechanism | Actively corrected by humans |
| Evaluation | Single snapshot | Ongoing tracking and improvement |

Key Insight:  
The baseline model achieves reasonable initial accuracy, but active human feedback targets the model's most confusing predictions, leading to gradual and measurable accuracy improvements.

## Why Accuracy Increases

- Human corrections specifically fix the areas where the model is least confident or wrong.
- These targeted corrections allow the model to adjust its decision boundaries more precisely.
- Retraining on a combination of original data and human-verified corrections leads to stronger, more robust performance on unseen validation samples.
- Over multiple retraining cycles, the model’s validation accuracy consistently improves.

## How to Run

1. Clone the repository
2. Install dependencies from requirements.txt
3. Create a virtual environment and activate it
4. Train the initial model by running:

```bash
python
>>> from model import train_and_save_model
>>> train_and_save_model("data/spam.csv")
```
5. Start the flask app : python app.py
6. Visit http://127.0.0.1:5000/ in your browser
7. Review, correct, retrain, and watch the model improve
8. Plot retrain progress using: python plot_retrain_progress.py

## Project Structure

```
hitl_sms_spam/
├── app.py                  # Flask app
├── model.py                 # Model training and loading
├── utils.py                 # Preprocessing functions
├── templates/
│   └── review.html          # Human review page
├── feedback/
│   └── feedback.json        # Human feedback storage
├── retrain_log.json         # Retrain logs
├── validation_set.json      # Fixed validation samples
├── plot_retrain_progress.py # Plotting retrain improvement
├── requirements.txt         # Python dependencies
└── data/
    └── spam.csv             # SMS Spam dataset
```

### *There are screenshots of the system (before styling the webpage, though the current version is styled)



