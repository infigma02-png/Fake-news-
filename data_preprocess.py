 
# data_preprocess.py
# import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_text(df):
    """
Cleans, balances, and splits the dataset into training and testing sets.
    """
    print(" Cleaning and balancing data...")

    # Drop rows with missing text or labels
    df = df.dropna(subset=['text', 'label'])

    # Shuffle dataset to avoid order bias
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Balance dataset (equal FAKE and REAL counts)
    label_counts = df['label'].value_counts()
    min_len = label_counts.min()
    df_balanced = df.groupby('label').head(min_len).reset_index(drop=True)

    print("Label distribution after balancing:")
    print(df_balanced['label'].value_counts(), "\n")

    # Feature and label
    X = df_balanced['text']
    y = df_balanced['label']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f" Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


