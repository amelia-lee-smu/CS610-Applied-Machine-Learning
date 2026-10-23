HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
NYC_TIMEZONE = "America/New_York"

# prediction 
NUM_RENTALS = 'num_rentals' # target variable
PREDICTION = 'prediction'

# location features:
STATION = "start station name"

# date features:
DATE = 'date'
WEEK_OF_YEAR = 'week_of_year'
SINE_WEEK_OF_YEAR = 'sine_week_of_year'
COSINE_WEEK_OF_YEAR = 'cosine_week_of_year'
DAY_OF_WEEK = 'day_of_week'
HOUR_OF_DAY = 'hour_of_day'
IS_PUBLIC_HOLIDAY = 'is_public_holiday'

# weather features (open-meteo hourly variables)
TEMPERATURE = 'temperature_2m'
RAIN = 'rain'
SNOW_DEPTH = 'snow_depth'
SNOWFALL = 'snowfall'
CLOUD_COVER_LOW = 'cloud_cover_low'
CLOUD_COVER_MID = 'cloud_cover_mid'
CLOUD_COVER_HIGH = 'cloud_cover_high'
RELATIVE_HUMIDITY = 'relative_humidity_2m'
PRECIPITATION = 'precipitation'
WIND_SPEED = 'wind_speed_10m'
WIND_GUSTS = 'wind_gusts_10m'
CLOUD_COVER = 'cloud_cover'

# intermediate features:
START_DATE = 'start_date'
END_DATE = 'end_date'
LATITUDE = 'start station latitude'
LONGITUDE = "start station longitude"

# final features
CATEGORICAL_COLUMNS = [STATION, HOUR_OF_DAY, DAY_OF_WEEK]
NUMERICAL_COLUMNS = [
    f'{NUM_RENTALS}_t-2', f'{NUM_RENTALS}_t-3', SINE_WEEK_OF_YEAR, COSINE_WEEK_OF_YEAR,
    TEMPERATURE, SNOW_DEPTH, CLOUD_COVER, CLOUD_COVER_MID, RELATIVE_HUMIDITY, PRECIPITATION, WIND_SPEED,
]
OTHER_COLUMNS = [IS_PUBLIC_HOLIDAY]

# expanding-window folds for cross-validation: (train start, train end, validation start, validation end)
VALIDATION_FOLDS = [
    ('2020-01-01', '2020-08-31', '2020-09-01', '2020-10-31'),
    ('2020-01-01', '2020-10-31', '2020-11-01', '2020-12-31'),
    ('2020-01-01', '2020-12-31', '2021-01-01', '2021-02-28'),
]

# fixed seed for Optuna
OPTUNA_SEED = 42
