import numpy as np
import pandas as pd
import torch
from torch import nn

from utilities.training import *
from utilities.utilities import *
from utilities.values import *


class PoissonMLP(nn.Module):
    '''
    The network itself.
    Station, hour and day of week are each turned into a small learned vector (an embedding),
    then joined with the numerical features and passed through fully connected layers.
    The output is the log of the expected rentals, so exp(output) is always positive.
    '''
    def __init__(self, n_stations, n_numerical, station_dim, hour_dim, day_dim, hidden_size, n_layers, dropout, initial_log_rate):
        super().__init__()
        self.station_embedding = nn.Embedding(n_stations, station_dim)
        self.hour_embedding = nn.Embedding(24, hour_dim)
        self.day_embedding = nn.Embedding(7, day_dim)

        layers = []
        input_size = station_dim + hour_dim + day_dim + n_numerical
        for _ in range(n_layers):
            layers += [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.layers = nn.Sequential(*layers)

        # start by predicting the average rentals, so training does not begin far from the answer
        nn.init.constant_(self.layers[-1].bias, initial_log_rate)

    def forward(self, station_index, hour, day, numerical):
        joined = torch.cat([
            self.station_embedding(station_index),
            self.hour_embedding(hour),
            self.day_embedding(day),
            numerical,
        ], dim = 1)
        return self.layers(joined).squeeze(1)


class NeuralNetworkModel:
    '''
    Preprocessing needed (fitted on only the train set):
        - Station: replaced by an index (the network learns a vector for each index)
        - Hour of day and day of week: used as indexes (the network learns a vector for each)
        - Numerical features: scaled
        - Other features (0/1 flags): used as they are
    The station index, scaler and network are stored together in this object, so they are saved and loaded together.
    The network is trained with the Poisson loss (negative log-likelihood), the same idea as the Poisson regression.
    '''
    def __init__(self, station_dim = 8, hidden_size = 64, n_layers = 2, dropout = 0.1,
                 learning_rate = 1e-3, weight_decay = 1e-5, epochs = 15, batch_size = 2048):
        # hyperparameters (tuned by Optuna)
        self.station_dim = station_dim        # size of the learned vector of each station
        self.hidden_size = hidden_size        # neurons in each hidden layer
        self.n_layers = n_layers              # number of hidden layers
        self.dropout = dropout                # share of neurons randomly switched off while training (higher = simpler)
        self.learning_rate = learning_rate    # size of each training step
        self.weight_decay = weight_decay      # strength of the L2 penalty on the weights
        self.epochs = epochs                  # number of passes over the train set
        self.batch_size = batch_size          # rows used for each training step

        # fixed settings
        self.hour_dim = 4     # size of the learned vector of each hour
        self.day_dim = 3      # size of the learned vector of each day of week

        # features used by this model
        self.numerical_cols = NUMERICAL_COLUMNS    # scaled
        self.other_cols = OTHER_COLUMNS            # used as they are (0/1 flags)
        self.target_col = NUM_RENTALS

        # scaler, station index and network (filled in by fit)
        self.scaler = ScalerProcessor()
        self.station_index = {}
        self.network = None

    def process_features(self, df):
        '''Returns the 4 inputs of the network, as tensors.'''
        # a station that was not in the train set gets index 0 (the "unknown" station)
        station_index = torch.tensor(df[STATION].map(self.station_index).fillna(0).astype(int).values, dtype = torch.long)
        hour = torch.tensor(df[HOUR_OF_DAY].values, dtype = torch.long)
        day = torch.tensor(df[DAY_OF_WEEK].values, dtype = torch.long)

        scaled = self.scaler.apply_scaler(df[self.numerical_cols])
        numerical = torch.tensor(np.hstack([scaled, df[self.other_cols].values]), dtype = torch.float32)
        return station_index, hour, day, numerical

    def fit(self, train_df):
        # fixed seed, so that the same data and hyperparameters give the same network
        torch.manual_seed(RANDOM_SEED)

        # fit the station index and the scaler on the train set (index 0 is kept for unknown stations)
        self.station_index = {station: i + 1 for i, station in enumerate(sorted(train_df[STATION].unique()))}
        self.scaler.fit(train_df[self.numerical_cols])

        station_index, hour, day, numerical = self.process_features(train_df)
        target = torch.tensor(train_df[self.target_col].values, dtype = torch.float32)

        self.network = PoissonMLP(
            n_stations = len(self.station_index) + 1,
            n_numerical = numerical.shape[1],
            station_dim = self.station_dim,
            hour_dim = self.hour_dim,
            day_dim = self.day_dim,
            hidden_size = self.hidden_size,
            n_layers = self.n_layers,
            dropout = self.dropout,
            initial_log_rate = float(np.log(train_df[self.target_col].mean())),
        )

        optimizer = torch.optim.AdamW(self.network.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        # the learning rate slowly goes down to 0 over the epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.epochs)
        loss_function = nn.PoissonNLLLoss(log_input = True)    # the network outputs log(expected rentals)

        n_rows = len(target)
        shuffler = torch.Generator().manual_seed(RANDOM_SEED)
        self.network.train()
        for epoch in range(self.epochs):
            # a new random order of the rows in each epoch
            order = torch.randperm(n_rows, generator = shuffler)
            for start in range(0, n_rows, self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                log_rate = self.network(station_index[batch], hour[batch], day[batch], numerical[batch])
                loss = loss_function(log_rate, target[batch])
                loss.backward()
                optimizer.step()
            scheduler.step()

    def predict(self, test_df):
        station_index, hour, day, numerical = self.process_features(test_df)

        self.network.eval()
        predictions = []
        with torch.no_grad():
            # in chunks, to keep the memory use small
            chunk_size = 65536
            for start in range(0, len(test_df), chunk_size):
                end = start + chunk_size
                log_rate = self.network(station_index[start:end], hour[start:end], day[start:end], numerical[start:end])
                predictions.append(torch.exp(log_rate.clamp(max = 10)).numpy())    # clamp: avoids overflow

        return np.concatenate(predictions).astype(float)

    def save_model(self, file_path):
        # Saves the whole object: station index, scaler, network and settings together
        save_to_pickle_file(self, file_path)

    @staticmethod
    def load_model(file_path):
        # Loads the whole object: station index, scaler, network and settings together
        return read_pickle_file(file_path)
