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


## Model training

Each model has its own [notebook](notebooks/) and [model file](utilities/). They use the same [training functions](utilities/training.py) and [cross-validation splits](utilities/values.py).

In each model's notebook, we do:
1. **Hyperparameter tuning with Optuna**
   - Optuna searches for the hyperparameters with the lowest mean Poisson deviance across the 3 folds.
   - MAE and RMSE are recorded for every trial as well.
3. **Prediction on the validation set (for the ensemble model)**
   - With the best hyperparameters, a new model is trained on each fold's train set.
   - The model is then used for prediction on the fold's validation set. The output is saved to `data/predictions/validation_<model>_fold<1-3>.parquet`.
4. **Final model and test predictions**
   - A new model is trained on the full train and validation set, and saved to `models/<model>_model.pkl`.
   - Predictions on the test sets is saved to `data/predictions/test_<model>.parquet`.


## Ensemble

Run [train_ensemble_colab.ipynb](notebooks/train_ensemble_colab.ipynb) on Google Colab after all 4 models are trained.

1. The ensemble model learns how to combine their saved validation and test predictions.
2. 4 ways of combining the models are compared: simple average, weighted average, weighted geometric average and stacking (a Poisson regression on the models' log predictions).
3. Each method is scored leave-one-fold-out: it learns its weights on 2 of the 3 validation folds and is scored on the third. The method with the lowest mean Poisson deviance is chosen.
4. The chosen method learns its final weights on all 3 validation folds, then combines the models' test predictions. Saved to `models/ensemble_model.pkl` and `data/predictions/test_ensemble.parquet`.
5. The test set is scored once, at the end of this notebook: each model, every ensemble method and the chosen ensemble.


## Team members (Group 1)

- Alessandro Pagan
- Alice Agazzi
- Lee Jing Hua Amelia
- Robin Thomas
- Suhasini Pillai Do Sivashankar
