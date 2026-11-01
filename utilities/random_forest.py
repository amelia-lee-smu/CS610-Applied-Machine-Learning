import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from utilities.training import *
from utilities.utilities import *
from utilities.values import *

# use the team's RANDOM_SEED (feat/lightgbm branch) if values.py has it
RANDOM_SEED = globals().get('RANDOM_SEED', 42)


class RandomForestModel:
    '''
    Preprocessing needed (fitted on only the train set):
        - Station name: converted to a number (ordinal encoded), since the trees need numbers
        - Hour of day and day of week: already numbers, used as they are
        - Numerical features: used as they are (trees split on thresholds, so scaling makes no difference)
    The encoder and model are stored together in this object, so they are saved and loaded together.
    '''
    def __init__(self, n_estimators = 200, max_depth = 20, min_samples_leaf = 5,
                 max_features = 0.5, max_samples = 0.3, criterion = 'poisson', random_state = RANDOM_SEED):
        # hyperparameters (tuned by Optuna)
        self.n_estimators = n_estimators            # number of trees (more = more stable, but slower)
        self.max_depth = max_depth                  # how deep each tree can grow (deeper = more complex)
        self.min_samples_leaf = min_samples_leaf    # fewest rows allowed in a leaf (higher = smoother predictions)
        self.max_features = max_features            # share of features each split may consider (lower = more varied trees)
        self.max_samples = max_samples              # share of the rows each tree is trained on (lower = faster)
        self.criterion = criterion                  # 'poisson' suits count targets; 'squared_error' is the default
        self.random_state = random_state            # fixed so the results are repeatable

        # features used by this model
        self.categorical_cols = CATEGORICAL_COLUMNS    # station is encoded; hour and day are already numbers
        self.numerical_cols = NUMERICAL_COLUMNS        # used as they are
        self.other_cols = OTHER_COLUMNS                # used as they are (0/1 flags)
        self.target_col = NUM_RENTALS

        # encoder and model
        # a station not seen in training is given -1, so prediction still works
        self.station_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
        self.model = RandomForestRegressor(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            max_features = self.max_features,
            max_samples = self.max_samples,
            criterion = self.criterion,
            random_state = self.random_state,
            n_jobs = -1,    # use every CPU core
        )

    def process_features(self, df):
        features_df = df[self.categorical_cols + self.numerical_cols + self.other_cols].copy()
        features_df[STATION] = self.station_encoder.transform(df[[STATION]]).ravel()
        return features_df

    def fit(self, train_df):
        # fit the encoder and model on the train set
        self.station_encoder.fit(train_df[[STATION]])

        processed_features = self.process_features(train_df)
        self.model.fit(processed_features, train_df[self.target_col])

    def predict(self, test_df):
        processed_features = self.process_features(test_df)
        return self.model.predict(processed_features)

    def feature_importances(self):
        # how much each feature reduced the error across all trees (adds up to 1)
        feature_names = self.categorical_cols + self.numerical_cols + self.other_cols
        return pd.Series(self.model.feature_importances_, index = feature_names).sort_values(ascending = False)

    def save_model(self, file_path):
        # Saves the whole object: encoder, model and settings together
        save_to_pickle_file(self, file_path)

    @staticmethod
    def load_model(file_path):
        # Loads the whole object: encoder, model and settings together
        return read_pickle_file(file_path)
