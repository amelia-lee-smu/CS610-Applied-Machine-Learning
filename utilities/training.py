'''
Functions and preprocessing classes shared by all models.
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error

from utilities.values import *
from utilities.utilities import *


def split_by_date(df, train_start, train_end, validation_start, validation_end):
    '''Splits the data into train and validation sets by date.'''
    train_df = df[(df[DATE] >= train_start) & (df[DATE] <= train_end)]
    validation_df = df[(df[DATE] >= validation_start) & (df[DATE] <= validation_end)]
    return train_df, validation_df


def evaluate(model, df):
    '''Scores the model's predictions. Returns MAE, RMSE and Poisson deviance.'''
    y_true = df[NUM_RENTALS]
    y_pred = model.predict(df)
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        # Poisson deviance cannot handle predictions of exactly 0 (possible for e.g. Random Forest),
        # so predictions are raised to a tiny positive value for this metric only
        'poisson_deviance': mean_poisson_deviance(y_true, np.clip(y_pred, 1e-8, None)),
    }


def cross_validate(model_class, hyperparams, train_validation_df, folds=VALIDATION_FOLDS):
    '''
    Trains and scores a new model on each validation fold. Returns one row of scores per fold.
    model_class: the model to create, e.g. PoissonRegressionModel
    hyperparams: its hyperparameters as a dict, e.g. {'alpha': 0.0001, 'solver': 'lbfgs'}
    '''
    results = []
    for fold_number, (train_start, train_end, validation_start, validation_end) in enumerate(folds, start = 1):
        train_df, validation_df = split_by_date(train_validation_df, train_start, train_end, validation_start, validation_end)
        print(f'Fold {fold_number}: training {model_class.__name__} on {len(train_df)} rows')

        # a new model for every fold, so nothing is carried over from a previous fold
        model = model_class(**hyperparams)
        model.fit(train_df)

        scores = evaluate(model, validation_df)
        scores['fold'] = fold_number
        results.append(scores)

    return pd.DataFrame(results)


def get_out_of_fold_predictions(model_class, hyperparams, train_validation_df, folds=VALIDATION_FOLDS):
    '''
    Predictions for every validation fold, using a new model trained only on the data before it. Used to learn the ensemble weights.
    model_class: the model to create, e.g. PoissonRegressionModel
        hyperparams: its hyperparameters as a dict, e.g. {'alpha': 0.0001, 'solver': 'lbfgs'}
    Returns a list of validation_dfs, each with a new prediction column.
    '''
    results = []
    for fold_number, (train_start, train_end, validation_start, validation_end) in enumerate(folds, start = 1):
        train_df, validation_df = split_by_date(train_validation_df, train_start, train_end, validation_start, validation_end)
        print(f'Fold {fold_number}: training {model_class.__name__} on {len(train_df)} rows')
        
        # a new model for every fold, so nothing is carried over from a previous fold
        model = model_class(**hyperparams)
        model.fit(train_df)

        validation_df[PREDICTION] = model.predict(validation_df)
        validation_df['fold'] = fold_number
        results.append(validation_df)

    return results


class OneHotEncoderProcessor:
    '''One-hot encoder (drops the first category of each feature, to avoid duplicating the intercept).'''
    def __init__(self):
        self.ohe = preprocessing.OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    def fit(self, features):
        self.ohe.fit(features)

    def apply_one_hot_encoder(self, features):
        return self.ohe.transform(features)


class ScalerProcessor:
    '''Standard scaler (rescales each feature to mean 0 and standard deviation 1).'''
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()

    def fit(self, features):
        self.scaler.fit(features)

    def apply_scaler(self, features):
        return self.scaler.transform(features)
