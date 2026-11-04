'''
Ensemble of the trained models.

Every base model has already made out-of-fold predictions on the 3 validation folds and predictions on the test set
(data/predictions/). The ensemble only learns how to combine those predictions, so the base models are not retrained.
'''
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error

from utilities.utilities import *
from utilities.values import *

KEY_COLUMNS = [STATION, DATE, HOUR_OF_DAY]
MIN_PREDICTION = 1e-6    # predictions are raised to this tiny value before taking logs or the Poisson deviance

ENSEMBLE_METHODS = {
    'mean': 'Simple average of the models',
    'weighted': 'Weighted average; the weights are learned and add up to 1',
    'geometric': 'Weighted geometric average (average of the log predictions); the weights add up to 1',
    'stacking': 'Poisson regression on the log predictions of the models (the weights do not need to add up to 1)',
}


def score(y_true, y_pred):
    '''MAE, RMSE and Poisson deviance, the same metrics as utilities/training.py.'''
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'poisson_deviance': mean_poisson_deviance(y_true, np.clip(y_pred, MIN_PREDICTION, None)),
    }


def read_predictions(file_paths):
    '''
    Reads one prediction file per model and puts them side by side: one column per model, one row per station-date-hour.
    file_paths: {model name: parquet file with a PREDICTION column}
    '''
    combined_df = None
    for model_name, file_path in file_paths.items():
        df = read_parquet_file(file_path)
        keep = KEY_COLUMNS + [NUM_RENTALS, PREDICTION] + (['fold'] if 'fold' in df.columns else [])
        df = df[keep].rename(columns = {PREDICTION: model_name})
        if combined_df is None:
            combined_df = df
        else:
            # every model must have predicted exactly the same rows, with the same actual rentals
            merged_df = combined_df.merge(df, on = [col for col in keep if col != PREDICTION], how = 'inner', validate = 'one_to_one')
            assert len(merged_df) == len(combined_df) == len(df), f'{model_name}: its rows do not match the other models'
            combined_df = merged_df
    return combined_df.reset_index(drop = True)


class EnsembleModel:
    '''
    Combines the predictions of several models into one prediction.
    method: one of ENSEMBLE_METHODS
    model_names: the prediction columns to combine
    '''
    def __init__(self, model_names, method = 'geometric'):
        assert method in ENSEMBLE_METHODS, f'method must be one of {list(ENSEMBLE_METHODS)}'
        self.model_names = list(model_names)
        self.method = method
        self.weights = np.full(len(self.model_names), 1 / len(self.model_names))
        self.stacker = None

    def _predictions(self, df):
        return np.clip(df[self.model_names].to_numpy(dtype = float), MIN_PREDICTION, None)

    def _combine(self, predictions, weights):
        if self.method == 'geometric':
            return np.exp(np.log(predictions) @ weights)
        return predictions @ weights

    def fit(self, df):
        '''df: out-of-fold predictions of the models (one column per model) and the actual rentals.'''
        predictions, y_true = self._predictions(df), df[NUM_RENTALS].to_numpy()

        if self.method in ('weighted', 'geometric'):
            # the weights are a softmax of free numbers, so they are always >= 0 and add up to 1
            to_weights = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
            loss = lambda z: mean_poisson_deviance(y_true, np.clip(self._combine(predictions, to_weights(z)), MIN_PREDICTION, None))
            result = minimize(loss, np.zeros(len(self.model_names)), method = 'Nelder-Mead',
                              options = {'maxiter': 4000, 'xatol': 1e-6, 'fatol': 1e-8})
            self.weights = to_weights(result.x)

        elif self.method == 'stacking':
            self.stacker = PoissonRegressor(alpha = 0, max_iter = 1000)
            self.stacker.fit(np.log(predictions), y_true)
            self.weights = self.stacker.coef_
        return self

    def predict(self, df):
        predictions = self._predictions(df)
        if self.method == 'stacking':
            return self.stacker.predict(np.log(predictions))
        return self._combine(predictions, self.weights)

    def weights_table(self):
        name = 'coefficient (log scale)' if self.method == 'stacking' else 'weight'
        return pd.Series(self.weights, index = self.model_names, name = name)

    def save_model(self, file_path):
        save_to_pickle_file(self, file_path)

    @staticmethod
    def load_model(file_path):
        return read_pickle_file(file_path)


def leave_one_fold_out(df, model_names, method):
    '''
    Honest validation score of an ensemble method: learn the combination on 2 of the 3 validation folds,
    score it on the fold that was left out, and repeat for each fold. Returns one row of scores per fold.
    '''
    rows = []
    for fold_number in sorted(df['fold'].unique()):
        train_df, held_out_df = df[df['fold'] != fold_number], df[df['fold'] == fold_number]
        ensemble = EnsembleModel(model_names, method).fit(train_df)
        rows.append({'method': method, 'fold': fold_number, **score(held_out_df[NUM_RENTALS], ensemble.predict(held_out_df))})
    return pd.DataFrame(rows)
