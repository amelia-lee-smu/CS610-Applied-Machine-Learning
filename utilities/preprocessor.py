from pathlib import Path

import holidays
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from utilities.values import *


class DatasetPreprocessor:
    '''
    Functions to process datetime.
    Note: no missing dates, so not handled
    '''
    def __init__(self):
        # get US public holidays
        self.holidays = holidays.US()

        # Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = retry_session)

    def read_all_files_in_dir(self, dir_paths):
        df = pd.DataFrame()
        for dir in dir_paths:
            # get all file names in the dir
            raw_files = [f for f in Path(dir).iterdir() if f.is_file()]

            # read and merge the files
            for file in raw_files: 
                curr_data = pd.read_csv(file)
                df = pd.concat([df, curr_data])
                
        return df

    # date extraction functions that require datetime
    def to_date(self, datetime_value):
        return datetime_value.date()

    def to_hour_of_day(self, datetime_value):
        return datetime_value.hour

    # date extraction functions that require date
    def to_year(self, date_value):
        return date_value.year

    def to_month(self, date_value):
        return date_value.month

    def to_week_of_year(self, date_value):
        return date_value.isocalendar().week

    def to_day_of_week(self, date_value):
        return date_value.weekday()

    def to_is_public_holiday(self, date_value):
        return 1 if date_value in self.holidays else 0

    def get_hourly_weather(self, params):
        # unpack params
        start_date, end_date, station, latitude, longitude = params[START_DATE], params[END_DATE], params[STATION], params[LATITUDE], params[LONGITUDE]

        # note: day_end has 1 extra day to handle daylight savings 
        day_start = pd.Timestamp(start_date, tz = NYC_TIMEZONE)
        day_end = pd.Timestamp(end_date, tz = NYC_TIMEZONE) + pd.DateOffset(days = 1)

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
        hourly_data_df[HOUR_OF_DAY] = hourly_data_df[DATE].apply(self.to_hour_of_day)
        hourly_data_df[DATE] = hourly_data_df[DATE].apply(self.to_date)
        hourly_data_df[STATION] = station
        hourly_data_df[LATITUDE] = latitude
        hourly_data_df[LONGITUDE] = longitude

        return hourly_data_df