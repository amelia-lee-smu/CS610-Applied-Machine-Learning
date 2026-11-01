import pickle
from pathlib import Path

import pandas as pd


def save_to_pickle_file(obj, file_path):
    # create the parent folder if it does not exist yet
    Path(file_path).parent.mkdir(parents = True, exist_ok = True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_to_parquet_file(df, file_path):
    # create the parent folder if it does not exist yet
    Path(file_path).parent.mkdir(parents = True, exist_ok = True)
    df.to_parquet(file_path, index = False)


def read_parquet_file(file_path):
    return pd.read_parquet(file_path)
