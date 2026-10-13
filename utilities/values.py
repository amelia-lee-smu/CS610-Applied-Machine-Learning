HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
NYC_TIMEZONE = "America/New_York"

# prediction 
NUM_RENTALS = 'num_rentals' # target variable

# location features:
STATION = "start station name"

# date features:
DATE = 'date'
YEAR = 'year'
MONTH = 'month'
WEEK_OF_YEAR = 'week_of_year'
DAY_OF_WEEK = 'day_of_week'
HOUR_OF_DAY = 'hour_of_day'
IS_PUBLIC_HOLIDAY = 'is_public_holiday'

# weather features (open-meteo hourly variables)
TEMPERATURE = 't-1_temperature_2m'
RAIN = 't-1_rain'
SNOW_DEPTH = 't-1_snow_depth'
SNOWFALL = 't-1_snowfall'
CLOUD_COVER_LOW = 't-1_cloud_cover_low'
CLOUD_COVER_MID = 't-1_cloud_cover_mid'
CLOUD_COVER_HIGH = 't-1_cloud_cover_high'
RELATIVE_HUMIDITY = 't-1_relative_humidity_2m'
PRECIPITATION = 't-1_precipitation'
WIND_SPEED = 't-1_wind_speed_10m'
WIND_GUSTS = 't-1_wind_gusts_10m'
CLOUD_COVER = 't-1_cloud_cover'

# intermediate features:
START_DATE = 'start_date'
END_DATE = 'end_date'
LATITUDE = 'start station latitude'
LONGITUDE = "start station longitude"