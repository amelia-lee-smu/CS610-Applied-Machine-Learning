from pathlib import Path

import holidays
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from sklearn import preprocessing

from utilities.values import *
from utilities.utilities import *

def read_all_files_in_dir(dir_paths):
    '''Reads all the csv files in a list of directories.'''
    df = pd.DataFrame()
    for dir in dir_paths:
        # get all file names in the dir
        raw_files = [f for f in Path(dir).iterdir() if f.is_file()]

        # read and merge the files
        for file in raw_files: 
            curr_data = pd.read_csv(file)
            df = pd.concat([df, curr_data])
            
    return df


class DateFeatureExtractor:
    '''
    Functions to process the datetime feature.
    '''
    def __init__(self):
        # get US public holidays
        self.holidays = holidays.US()

        self.max_weeks_in_year = 53

    def to_date(self, datetime_value):
        return datetime_value.date()

    def to_hour_of_day(self, datetime_value):
        return datetime_value.hour

    def to_week_of_year(self, date_value):
        return date_value.isocalendar().week

    def to_sine_week_of_year(self, week_of_year):
        week_angle = 2 * np.pi * (week_of_year - 1) / self.max_weeks_in_year
        return np.sin(week_angle)
    
    def to_cosine_week_of_year(self, week_of_year):
        week_angle = 2 * np.pi * (week_of_year - 1) / self.max_weeks_in_year
        return np.cos(week_angle)

    def to_day_of_week(self, date_value):
        return date_value.weekday()

    def to_is_public_holiday(self, date_value):
        return 1 if date_value in self.holidays else 0


class WeatherFeatureExtractor:
    '''
    Functions to call the weather API and process the results.
    '''
    def __init__(self, start_date, end_date):
        # Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = retry_session)

        self.start_date = start_date
        self.end_date = end_date

        # needed to process the hourly data timestamp 
        self.date_feature_extractor = DateFeatureExtractor()

    def get_hourly_weather(self, station, latitude, longitude):
        # note: day_end has 1 extra day to handle daylight savings 
        day_start = pd.Timestamp(self.start_date, tz = NYC_TIMEZONE)
        day_end = pd.Timestamp(self.end_date, tz = NYC_TIMEZONE) + pd.DateOffset(days = 1)

        # query
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": day_start.strftime('%Y-%m-%d'),
            "end_date": day_end.strftime('%Y-%m-%d'),   
            "hourly": [TEMPERATURE, RAIN, SNOW_DEPTH, SNOWFALL, CLOUD_COVER_LOW, CLOUD_COVER_MID, CLOUD_COVER_HIGH, RELATIVE_HUMIDITY, PRECIPITATION, WIND_SPEED, WIND_GUSTS, CLOUD_COVER],
            "timezone": "GMT",
        }
        responses = self.openmeteo.weather_api(HISTORICAL_WEATHER_URL, params = params)

        # format response
        hourly = responses[0].Hourly()
        hourly_data = {
            DATE: pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            ).tz_convert(NYC_TIMEZONE)
        }

        hourly_data[TEMPERATURE] = hourly.Variables(0).ValuesAsNumpy()
        hourly_data[RAIN] = hourly.Variables(1).ValuesAsNumpy()
        hourly_data[SNOW_DEPTH] = hourly.Variables(2).ValuesAsNumpy()
        hourly_data[SNOWFALL] = hourly.Variables(3).ValuesAsNumpy()
        hourly_data[CLOUD_COVER_LOW] = hourly.Variables(4).ValuesAsNumpy()
        hourly_data[CLOUD_COVER_MID] = hourly.Variables(5).ValuesAsNumpy()
        hourly_data[CLOUD_COVER_HIGH] = hourly.Variables(6).ValuesAsNumpy()
        hourly_data[RELATIVE_HUMIDITY] = hourly.Variables(7).ValuesAsNumpy()
        hourly_data[PRECIPITATION] = hourly.Variables(8).ValuesAsNumpy()
        hourly_data[WIND_SPEED] = hourly.Variables(9).ValuesAsNumpy()
        hourly_data[WIND_GUSTS] = hourly.Variables(10).ValuesAsNumpy()
        hourly_data[CLOUD_COVER] = hourly.Variables(11).ValuesAsNumpy()
        hourly_data_df = pd.DataFrame(data = hourly_data)

        # keep only the requested local days (handling of daylight savings)
        hourly_data_df = hourly_data_df[
            (hourly_data_df[DATE] >= day_start) & (hourly_data_df[DATE] < day_end)
        ].reset_index(drop = True)

        # formatting to match back to the full dataset
        hourly_data_df[HOUR_OF_DAY] = hourly_data_df[DATE].apply(self.date_feature_extractor.to_hour_of_day)
        hourly_data_df[DATE] = hourly_data_df[DATE].apply(self.date_feature_extractor.to_date)
        hourly_data_df[STATION] = station
        hourly_data_df[LATITUDE] = latitude
        hourly_data_df[LONGITUDE] = longitude

        return hourly_data_df


class FeaturePreprocessor:
    '''
    Functions for one-hot encoding and scaling. 
    '''
    def __init__(self):
        # one-hot encoder and scaler
        self.ohe = preprocessing.OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        self.scaler = preprocessing.StandardScaler()

    def fit_one_hot_encoder(self, categorical_features):
        self.ohe.fit(categorical_features)

    def get_encoded_features_df(self, df, categorical_cols, non_categorical_cols):
        non_categorical_features = df[non_categorical_cols]
        categorical_features = pd.DataFrame(
            self.ohe.transform(df[categorical_cols]),
            columns=self.ohe.get_feature_names_out(categorical_cols),
            index=df.index,
        )
        combined_features_df = pd.concat([categorical_features, non_categorical_features], axis=1)
        return combined_features_df

    def save_one_hot_encoder(self, file_path):
        save_to_pickle_file(self.ohe, file_path)

    def load_one_hot_encoder(self, file_path):
        self.ohe = read_pickle_file(file_path)

    def fit_scaler(self, df):
        self.scaler.fit(df)

    def get_scaled_features(self, df):
        scaled_df = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.scaler.get_feature_names_out(),
            index=df.index,
        )
        return scaled_df
