# CS610 Applied Machine Learning - Group 1

Hourly rental demand forecasting for 51 NYC Citi Bike stations (Jan 2020 – Apr 2021).

We forecast the number of rental starts at each station in the next hour. Four base models (Poisson Regression,
Random Forest, LightGBM and a neural network) and four ensemble approaches were compared against two historical naive baselines. The **weighted geometric mean ensemble** model was selected as the final model.

## Installation

```bash
conda env create -f environment.yml
conda activate applied-ml
pip install openmeteo-requests requests-cache retry-requests   # for weather API
```

## 1. Data preprocessing

Run [notebooks/data_preprocessing.ipynb](notebooks/data_preprocessing.ipynb) from top to bottom.

1. **Rental counts**
   - Duplicate trips are removed, and the depot (`JCBS Depot`) is excluded.
   - Trips are counted per station per hour, on a complete station × date × hour grid. Hours with no trips are 0.
2. **Features**
   - We assume that forecasts are made 1 hour prior. For example, 12:00's demand is forecasted at 11:00.
   - Date: week of year, day of week, hour of day, US public holiday flag.
   - Previous rental demand: T-2 and T-3 complete hourly rental counts (i.e. the 09:00 to 11:00).
   - Hourly weather features from the [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api):
     T-1 values (i.e. the observed weather features at 11:00).
3. **Train-validation and test split**
   - Train-validation: 1 Jan 2020 – 28 Feb 2021, used for 3-folds time series cross-validation.
   - Test: 1 Mar – 30 Apr 2021, used only for the final evaluation.

## 2. Model training

Each model has its own notebook and [model file](utilities/). They use the same
[training functions](utilities/training.py) and the same 3 cross-validation folds:

| Fold | Train | Validation |
|---|---|---|
| 1 | Jan – Aug 2020 | Sep – Oct 2020 |
| 2 | Jan – Oct 2020 | Nov – Dec 2020 |
| 3 | Jan – Dec 2020 | Jan – Feb 2021 |

In each notebook, we do:

1. **Hyperparameter tuning with Optuna**
   - Optuna searches for the hyperparameters with the lowest mean Poisson deviance across the 3 folds.
   - MAE and RMSE are recorded for every trial as well.
2. **Validation predictions (for the ensemble)**
   - With the best hyperparameters, a new model is trained on each fold's train set and predicts that fold's
     validation set.
3. **Final model and test predictions**
   - A new model is trained on the full train-validation set and saved.
   - Its test predictions are saved too.

## 3. Ensemble
4 approaches were tested to combine the base learner predictions: 
| Approach | Description |
|---|---|
| Equal arithmetic mean | Simple average, so every model counts equally (0.25 each) |
| Weighted arithmetic mean | Weighted average; the weights are learned, non-negative and sum to 1 |
| Weighted geometric mean (selected) | Weighted average on the log scale; the weights are learned, non-negative and sum to 1 |
| Poisson stacking | Poisson regression on the models' log predictions; the weights need not sum to 1 |

## 4. Results
Only the selected model and the baselines are displayed here for brevity.
Against the 4-week average, the ensemble's test MAE is 1.7% lower and its RMSE is 8.7% lower. This suggests that it has corrected the larger errors. 

| Method | Validation Poisson deviance (Folds 2–3) | Test MAE | Test RMSE | Test Poisson deviance |
|---|---|---|---|---|
| Ensemble (weighted geometric mean), selected | 0.7359 | 0.5027 | 0.9492 | 0.9403 |
| Baseline (4-week average) | – | 0.5112 | 1.0401 | 1.7994 |
| Baseline (previous week) | – | 0.6069 | 1.3002 | 3.6766 |
<br>

However, the benefits are more pronounced where demand matters the most:

| Test subset | Station-hours | MAE vs 4-week average | RMSE vs 4-week average |
|---|---|---|---|
| Non-zero hours| 21,145 | 7.3% lower | 7.7% lower |
| Top 10 busiest stations | 14,580 | 8.2% lower | 10.5% lower |
| Weekend peaks (12:00–18:00) | 4,896 | 14.5% lower | 17.2% lower |
| Weekday peaks (07:00–09:00, 16:00–20:00) | 13,770 | 0.2% lower | 5.5% lower |

The full table for every model and ensemble method is produced by
[notebooks/results_reporting.ipynb](notebooks/results_reporting.ipynb).

## Team members (Group 1)
- Alessandro Pagan
- Alice Agazzi
- Lee Jing Hua Amelia
- Robin Thomas
- Suhasini Pillai Do Sivashankar
