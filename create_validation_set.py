# create_validation_set.py

import pandas as pd
import json
from sklearn.model_selection import train_test_split
from utils import load_and_preprocess_data

def create_fixed_validation_set():
    # Load original data
    df = load_and_preprocess_data("data/spam.csv")

    # Split into 80% train, 20% validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Save validation set
    validation_set = {
        "texts": val_df['text'].tolist(),
        "labels": val_df['label'].tolist()
    }

    with open("validation_set.json", "w") as f:
        json.dump(validation_set, f, indent=4)

    print(f"✅ Validation set created and saved! ({len(val_df)} samples)")

if __name__ == "__main__":
    create_fixed_validation_set()
