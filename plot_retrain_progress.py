import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_retrain_progress():
    # Load retrain log
    with open('retrain_log.json', 'r') as f:
        log_data = json.load(f)

    df = pd.DataFrame(log_data)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(df['feedback_count'], df['initial_accuracy'], marker='o', label='Initial Accuracy')
    plt.plot(df['feedback_count'], df['retrained_accuracy'], marker='o', label='Retrained Accuracy')
    plt.xlabel('Feedback Corrections')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Improvement Over Retrains')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_retrain_progress()
