# Drop-in helper: from utilities.nonzero_eval import score_all_vs_nonzero
# Usage in any model notebook:  score_all_vs_nonzero(y_test, {'lightgbm': y_pred})
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance

def score_all_vs_nonzero(y_true, preds, eps=1e-3):
    """Score one or more models on ALL rows and on rows where actual rentals > 0 only.

    y_true : actual rentals (array/Series)
    preds  : dict {model_name: predictions}  (same length/order as y_true)
    Returns a DataFrame indexed by (subset, model) with MAE, RMSE, Poisson deviance and n.
    Predictions are clipped at `eps` because Poisson deviance needs predictions > 0.
    """
    y = np.asarray(y_true, float)
    rows = []
    for subset, mask in {"all rows": np.ones(len(y), bool), "actual > 0 only": y > 0}.items():
        for name, p in preds.items():
            p = np.clip(np.asarray(p, float)[mask], eps, None)
            rows.append({"subset": subset, "model": name,
                         "MAE": mean_absolute_error(y[mask], p),
                         "RMSE": mean_squared_error(y[mask], p) ** 0.5,
                         "Poisson dev": mean_poisson_deviance(y[mask], p),
                         "n": int(mask.sum())})
    return pd.DataFrame(rows).set_index(["subset", "model"])
