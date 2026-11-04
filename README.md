# CS610 Applied Machine Learning - Group 1

Hourly rental demand forecasting for 51 NYC Citi Bike stations (Jan 2020 – Apr 2021).

## Installation

```bash
conda env create -f environment.yml
conda activate applied-ml
pip install openmeteo-requests requests-cache retry-requests   # for weather API
```

## Data preprocessing

Run [notebooks/data_preprocessing.ipynb](notebooks/data_preprocessing.ipynb) from top to bottom. A brief summary of the steps taken:

1. **Rental counts**
   - Trips are counted per station per hour, on a complete station × date × hour grid. Hours with no trips are 0.
   - The depot (`JCBS Depot`) is excluded.
2. **Features**
   - Note: we are assuming that the prediction will be done 1 hour earlier. For example, the prediction of 12 noon's demand will be done at 11am. 
   - Date: week of year, day of week, hour of day, US public holiday flag.
   - Previous rental demand: T-2 and T-3 actual rental counts are used (i.e. 9am and 10am rental numbers).
   - Hourly weather data from the [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api): T-1 data is used (i.e. the latest data at 11am).
3. **Train-Validation and Test split**
   - Train-validation: 1 Jan 2020 – 28 Feb 2021 (combining for cross-validation in model training).
   - Test: 1 Mar – 30 Apr 2021
   - Processed data is stored in the [data/processed/](https://github.com/amelia-lee-smu/CS610-Applied-Machine-Learning/tree/feat/eda-for-proposal/data/processed) folder. 


## Team members (Group 1)

- Alessandro Pagan
- Alice Agazzi
- Lee Jing Hua Amelia
- Robin Thomas
- Suhasini Pillai Do Sivashankar
