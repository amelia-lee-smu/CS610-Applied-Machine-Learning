from abc import ABC, abstractmethod

import numpy as np
import optuna
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error

from utilities.values import *
from utilities.utilities import *


class BaseModel(ABC):
    '''
    Code shared by all models: scaling (if any), training, evaluation, cross-validation and saving.
    Each model only needs to define create_model(), which returns a new, untrained model.
    '''
    def __init__(self, use_scaler, numeric_cols = NUMERIC_COLS):
        # use_scaler: True for models that need scaled features (e.g. Poisson regression, MLP),
        #             False for tree-based models (e.g. Random Forest, LightGBM)
        self.use_scaler = use_scaler
        self.numeric_cols = numeric_cols
        self.scaler = None
        self.model = None

    @abstractmethod
    def create_model(self):
        '''Returns a new, untrained model.'''

    @staticmethod
    def suggest_hyperparams(trial):
        '''
        Returns the hyperparameters to try in an Optuna trial, as a dict.
        Each model defines its own search space; the keys must match the model's __init__ arguments.
        '''
        raise NotImplementedError

    @classmethod
    def tune(cls, train_validation_df, n_trials = 30, folds = VALIDATION_FOLDS):
        '''
        Uses Optuna to search for the hyperparameters with the lowest mean Poisson deviance
        across the validation folds. Returns the Optuna study (see study.best_params).
        '''
        def objective(trial):
            model = cls(**cls.suggest_hyperparams(trial))
            fold_scores = []
            for fold, dates in enumerate(folds, start = 1):
                train_df, validation_df = split_by_date(train_validation_df, *dates)
                model.fit(train_df)
                fold_scores.append(model.evaluate(validation_df)['poisson_deviance'])

                # stop unpromising trials early to save time
                trial.report(np.mean(fold_scores), step = fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return np.mean(fold_scores)

        study = optuna.create_study(direction = 'minimize', pruner = optuna.pruners.MedianPruner(n_startup_trials = 5))
        study.optimize(objective, n_trials = n_trials)
        return study

    def get_feature_cols(self, df):
        # every column except the target and the date
        return [col for col in df.columns if col not in [NUM_RENTALS, DATE]]

    def get_features(self, df):
        features_df = df[self.get_feature_cols(df)].copy()
        # apply scaler on features if needed
        if self.use_scaler:
            features_df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return features_df

    def fit(self, train_df):
        # start from a new scaler and model, so nothing is carried over from a previous fold
        if self.use_scaler:
            self.scaler = preprocessing.StandardScaler()
            self.scaler.fit(train_df[self.numeric_cols])
        self.model = self.create_model()
        self.model.fit(self.get_features(train_df), train_df[NUM_RENTALS])

    def predict(self, df):
        return self.model.predict(self.get_features(df))

    def evaluate(self, df):
        y_true = df[NUM_RENTALS]
        y_pred = self.predict(df)
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': root_mean_squared_error(y_true, y_pred),
            # Poisson deviance is undefined for predictions of exactly 0 (possible for e.g. Random Forest),
            # so predictions are raised to a tiny positive value for this metric only
            'poisson_deviance': mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None)),
        }

    def cross_validate(self, train_validation_df, folds = VALIDATION_FOLDS):
        '''Trains and evaluates the model on each fold. Returns one row of scores per fold.'''
        results = []
        for fold, dates in enumerate(folds, start = 1):
            train_df, validation_df = split_by_date(train_validation_df, *dates)
            self.fit(train_df)
            results.append({'fold': fold, **self.evaluate(validation_df)})
        return pd.DataFrame(results)

    def save_model(self, file_path):
        # the scaler is saved with the model, since predictions need both
        save_to_pickle_file({'scaler': self.scaler, 'model': self.model, 'numeric_cols': self.numeric_cols}, file_path)

    def load_model(self, file_path):
        saved = read_pickle_file(file_path)
        self.scaler = saved['scaler']
        self.model = saved['model']
        self.numeric_cols = saved['numeric_cols']
