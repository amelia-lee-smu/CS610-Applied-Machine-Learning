import pandas as pd
from lightgbm import LGBMRegressor

from utilities.training import *
from utilities.utilities import *
from utilities.values import *


class LightGBMModel:
    '''
    Preprocessing needed (fitted on only the train set):
        - Categorical features (station, hour, day of week): stored as pandas categories, which LightGBM splits on directly
          (no one-hot encoding needed)
        - Everything else is used as it is: trees do not need scaling
    The list of categories and the model are stored together in this object, so they are saved and loaded together.
    The model uses a Poisson objective (it predicts the expected number of rentals, always >= 0).
    '''
    def __init__(self, learning_rate = 0.05, n_estimators = 500, num_leaves = 63, min_child_samples = 50,
                 subsample = 0.8, colsample_bytree = 0.8, reg_lambda = 1.0):
        # hyperparameters (tuned by Optuna)
        self.learning_rate = learning_rate          # how much each new tree adds (lower = slower, safer learning)
        self.n_estimators = n_estimators            # number of boosting rounds (trees)
        self.num_leaves = num_leaves                # maximum leaves in one tree (higher = more complex)
        self.min_child_samples = min_child_samples  # smallest number of rows allowed in a leaf (higher = simpler)
        self.subsample = subsample                  # share of the rows used for each tree
        self.colsample_bytree = colsample_bytree    # share of the features used for each tree
        self.reg_lambda = reg_lambda                # strength of the L2 penalty on the leaf values

        # features used by this model
        self.categorical_cols = CATEGORICAL_COLUMNS    # LightGBM categories
        self.numerical_cols = NUMERICAL_COLUMNS        # used as they are (not scaled)
        self.other_cols = OTHER_COLUMNS                # used as they are (0/1 flags)
        self.target_col = NUM_RENTALS

        # the categories seen in the train set (filled in by fit)
        self.categories = {}

        self.model = LGBMRegressor(
            objective = 'poisson',
            learning_rate = self.learning_rate,
            n_estimators = self.n_estimators,
            num_leaves = self.num_leaves,
            min_child_samples = self.min_child_samples,
            subsample = self.subsample,
            subsample_freq = 1,     # needed for subsample to have any effect
            colsample_bytree = self.colsample_bytree,
            reg_lambda = self.reg_lambda,
            n_jobs = -1,
            random_state = RANDOM_SEED,
            verbose = -1,
        )

    def process_features(self, df):
        features = df[self.categorical_cols + self.numerical_cols + self.other_cols].copy()
        # a value that was not in the train set becomes missing (NaN), which LightGBM handles
        for col in self.categorical_cols:
            features[col] = pd.Categorical(features[col], categories = self.categories[col])
        return features

    def fit(self, train_df):
        # remember the categories of the train set, then fit the model
        self.categories = {col: sorted(train_df[col].unique()) for col in self.categorical_cols}

        processed_features = self.process_features(train_df)
        self.model.fit(processed_features, train_df[self.target_col], categorical_feature = self.categorical_cols)

    def predict(self, test_df):
        processed_features = self.process_features(test_df)
        return self.model.predict(processed_features)

    def save_model(self, file_path):
        # Saves the whole object: categories, model and settings together
        save_to_pickle_file(self, file_path)

    @staticmethod
    def load_model(file_path):
        # Loads the whole object: categories, model and settings together
        return read_pickle_file(file_path)
