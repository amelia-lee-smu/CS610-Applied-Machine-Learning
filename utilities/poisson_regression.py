import pandas as pd
from sklearn.linear_model import PoissonRegressor

from utilities.training import *
from utilities.utilities import *
from utilities.values import *


class PoissonRegressionModel:
    '''
    Preprocessing needed (fitted on only the train set):
        - Categorical features: one-hot encoded 
        - Numerical features: scaled
    The encoder, scaler and model are stored together in this object, so they are saved and loaded together.
    '''
    def __init__(self, alpha = 1e-4, solver = 'lbfgs'):
        # hyperparameters (tuned by Optuna)
        self.alpha = alpha      # strength of the L2 penalty (higher = simpler model)
        self.solver = solver    # algorithm used to fit the model

        # features used by this model
        self.categorical_cols = CATEGORICAL_COLUMNS    # one-hot encoded
        self.numerical_cols = NUMERICAL_COLUMNS        # scaled
        self.other_cols = OTHER_COLUMNS                # used as they are (0/1 flags)
        self.target_col = NUM_RENTALS

        # encoder, scaler and model
        self.one_hot_encoder = OneHotEncoderProcessor()
        self.scaler = ScalerProcessor()
        self.model = PoissonRegressor(alpha = self.alpha, solver = self.solver, max_iter = 500)

    def process_features(self, df):
        # One-hot encode categorical features
        one_hot_df = pd.DataFrame(
            self.one_hot_encoder.apply_one_hot_encoder(df[self.categorical_cols]),
            columns = self.one_hot_encoder.ohe.get_feature_names_out(self.categorical_cols),
            index = df.index,
        )

        # Scale the numerical features
        scaled_df = pd.DataFrame(
            self.scaler.apply_scaler(df[self.numerical_cols]),
            columns = self.numerical_cols,
            index = df.index,
        )

        # return all the features, including the untouched columns
        return pd.concat([one_hot_df, scaled_df, df[self.other_cols]], axis = 1)

    def fit(self, train_df):
        # fit the encoder, scaler, and model on the train set
        self.one_hot_encoder.fit(train_df[self.categorical_cols])
        self.scaler.fit(train_df[self.numerical_cols])

        processed_features = self.process_features(train_df)
        self.model.fit(processed_features, train_df[self.target_col])

    def predict(self, test_df):
        processed_features = self.process_features(test_df)
        return self.model.predict(processed_features)

    def save_model(self, file_path):
        # Saves the whole object: encoder, scaler, model and settings together
        save_to_pickle_file(self, file_path)

    @staticmethod
    def load_model(file_path):
        # Loads the whole object: encoder, scaler, model and settings together
        return read_pickle_file(file_path)
