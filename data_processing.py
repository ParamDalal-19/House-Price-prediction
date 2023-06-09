import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def split_train_test(data, test_size=0.2, random_state=42):
    train_set, test_set = train_test_split(
        data, test_size=test_size, random_state=random_state)
    return train_set, test_set
