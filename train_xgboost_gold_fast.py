"""Train XGBoost on Min San's Citi Bike Gold feature_store.zip + label_store.zip.

From the MLE repository root:
    python -m pip install -r requirements.txt
    python scripts/train_xgboost_gold.py \
      --features-zip data/feature_store.zip --labels-zip data/label_store.zip

Optional retrospective experiment (see README re weather leakage):
    python scripts/train_xgboost_gold.py --features-zip data/feature_store.zip \
      --labels-zip data/label_store.zip --include-weather

Only committed code/report should go to GitHub; raw datasets and model are ignored.
"""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

FEATURE_PATTERN = re.compile(
    r"(?:^|/)feature_store/gold_feature_store_(\d{4}_\d{2}_\d{2})\.parquet/part-[^/]+\.snappy\.parquet$"
)
LABEL_PATTERN = re.compile(
    r"(?:^|/)label_store/gold_label_store_(\d{4}_\d{2}_\d{2})\.parquet/part-[^/]+\.snappy\.parquet$"
)
KEYS = ["station_id", "snapshot_date"]
NUMERIC_COLS = [
    "fe_lag_1", "fe_lag_7", "fe_lag_14", "fe_rolling_mean_7",
    "fe_rolling_mean_28", "fe_day_of_week", "fe_is_weekend", "fe_month",
]
WEATHER_COLS = [
    "mean_temperature_c", "med_temperature_c", "max_temperature_c",
    "min_temperature_c", "temperature_range_c", "total_precipitation_mm",
    "total_rain_mm", "mean_cloud_pct", "mean_wind_speed_kmh",
    "max_wind_speed_kmh", "wet_hours",
]


def load_parquet_source(source_path: Path, kind: str) -> pd.DataFrame:
    """Read Spark daily Parquet partitions from an extracted folder OR original ZIP."""
    pattern = FEATURE_PATTERN if kind == "feature" else LABEL_PATTERN
    frames = []
    by_date = Counter()
    ignored_legacy = 0
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with ZipFile(source_path) as zf:
            matches = [name for name in zf.namelist() if pattern.search(name)]
            if not matches:
                raise ValueError(f"No {kind} Spark Parquet files in ZIP: {source_path}")
            for i, name in enumerate(matches, 1):
                blob = zf.read(name)
                if kind == "feature" and b'"mean_temperature_c"' not in blob:
                    ignored_legacy += 1
                    continue
                frames.append(pd.read_parquet(BytesIO(blob), engine="pyarrow"))
                by_date[pattern.search(name).group(1)] += 1
                if i % 1000 == 0:
                    print(f"  {kind}: inspected {i}/{len(matches)} partitions...", flush=True)
    elif source_path.is_dir():
        matches = [p for p in source_path.rglob("part-*.snappy.parquet")
                   if pattern.search(p.as_posix())]
        if not matches:
            raise ValueError(f"No {kind} Spark Parquet files in folder: {source_path}")
        for i, path in enumerate(matches, 1):
            # Spark files include the schema at the end of the Parquet file.
            if kind == "feature":
                with path.open("rb") as handle:
                    handle.seek(max(0, path.stat().st_size - 5000))
                    footer = handle.read()
                if b'"mean_temperature_c"' not in footer:
                    ignored_legacy += 1
                    continue
            frames.append(pd.read_parquet(path, engine="pyarrow"))
            by_date[pattern.search(path.as_posix()).group(1)] += 1
            if i % 1000 == 0:
                print(f"  {kind}: inspected {i}/{len(matches)} partitions...", flush=True)
    else:
        raise FileNotFoundError(f"Cannot find {kind} input folder or ZIP: {source_path}")
    if not frames:
        raise ValueError(f"No usable {kind} Parquet partitions found: {source_path}")
    result = pd.concat(frames, ignore_index=True)
    del frames
    print(f"Loaded {kind}: {len(result):,} rows across {len(by_date)} dates; "
          f"skipped {ignored_legacy} older weatherless files", flush=True)
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="raise").dt.normalize()
    result["station_id"] = result["station_id"].astype("string")
    if result[KEYS].isna().any().any():
        raise ValueError(f"Missing station_id/date join keys in {kind} dataset")
    duplicates = result.duplicated(KEYS, keep=False)
    if duplicates.any():
        eg = result.loc[duplicates, KEYS].head(5).to_dict("records")
        raise ValueError(f"Duplicate station-date keys in {kind}: {int(duplicates.sum())} rows. Example: {eg}")
    return result


def prepare_data(feature_zip: Path, label_zip: Path, include_weather: bool) -> tuple[pd.DataFrame, list[str]]:
    feat = load_parquet_source(feature_zip, "feature")
    labels = load_parquet_source(label_zip, "label")
    required = set(KEYS + NUMERIC_COLS)
    if not required.issubset(feat.columns):
        raise ValueError(f"Missing feature columns: {sorted(required - set(feat.columns))}")
    if not set(KEYS + ["label", "label_def"]).issubset(labels.columns):
        raise ValueError(f"Unexpected label columns: {labels.columns.tolist()}")
    definitions = labels["label_def"].dropna().unique().tolist()
    print("Target definitions:", definitions)
    if definitions != ["trip_starts_1d"]:
        raise ValueError("Target is not exclusively trip_starts_1d; confirm label semantics before training")
    if include_weather:
        missing = sorted(set(WEATHER_COLS) - set(feat.columns))
        if missing:
            raise ValueError(f"Weather columns missing: {missing}")
        print("WARNING: Using OBSERVED weather. Not a valid 06:00 forecast evaluation "
              "unless weather was available as of the prediction cutoff.")
    columns = [*NUMERIC_COLS, *(WEATHER_COLS if include_weather else [])]
    merged = feat.merge(
        labels[KEYS + ["label"]], on=KEYS, how="inner", validate="one_to_one"
    )
    if len(merged) < len(labels):
        print(f"WARNING: {len(labels)-len(merged):,} label rows did not join to features")
    if len(merged) == 0:
        raise ValueError("No joined training records. Check station_id and snapshot_date types")
    merged["label"] = pd.to_numeric(merged["label"], errors="coerce")
    bad_target = merged["label"].isna() | (merged["label"] < 0)
    if bad_target.any():
        raise ValueError(f"Found {int(bad_target.sum())} missing or negative targets")
    for col in columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["station_id"] = merged["station_id"].astype(str)
    print("Joined rows:", f"{len(merged):,}", "Unique stations:", merged["station_id"].nunique())
    print("Date range:", merged.snapshot_date.min().date(), "to", merged.snapshot_date.max().date())
    return merged, ["station_id", *columns]


def splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = df["snapshot_date"]
    out = {
        "train": df[d < "2017-01-01"],
        "validation": df[(d >= "2017-01-01") & (d < "2017-04-01")],
        "test": df[(d >= "2017-04-01") & (d < "2017-07-01")],
        "out_of_time": df[d >= "2017-07-01"],
    }
    for label, x in out.items():
        print(f"{label}: {len(x):,} rows")
        if len(x) == 0:
            raise ValueError(f"No rows in {label}; verify dataset coverage and date splits")
    return out


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    pred = np.maximum(0, np.asarray(prediction, dtype=float))
    # Poisson deviance requires strictly positive predictions.
    pred_for_poisson = np.maximum(1e-9, pred)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
        "mean_poisson_deviance": float(mean_poisson_deviance(y, pred_for_poisson)),
    }


def fit_model(df: pd.DataFrame, features: list[str], model_dir: Path, report_dir: Path, include_weather: bool) -> None:
    partitions = splits(df)
    numeric = [c for c in features if c != "station_id"]
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("station", OneHotEncoder(handle_unknown="ignore"), ["station_id"]),
        ],
        remainder="drop",
    )
    estimator = XGBRegressor(
        objective="count:poisson",
        tree_method="hist",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
    )
    model = Pipeline([("preprocess", preprocessing), ("xgb", estimator)])
    train = partitions["train"]
    model.fit(train[features], train["label"].to_numpy())
    results = {}
    for name, part in partitions.items():
        y = part["label"].to_numpy()
        pred = model.predict(part[features])
        results[name] = metrics(y, pred)
        print(f"{name}:", {k: round(v, 4) for k, v in results[name].items()})
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "xgboost_gold_pipeline.joblib")
    model.named_steps["xgb"].save_model(model_dir / "xgboost_gold_estimator.json")
    report = {
        "target": "trip_starts_1d", "join_keys": KEYS,
        "included_observed_weather": include_weather,
        "features": features,
        "split_dates": {
            "train": "2016-01-01 to 2016-12-31",
            "validation": "2017-01-01 to 2017-03-31",
            "test": "2017-04-01 to 2017-06-30",
            "out_of_time": "2017-07-01 to 2017-12-31",
        },
        "metrics": results,
        "caveat": ("Observed weather may not be known at forecast cutoff; validate availability "
                   "before deploying a next-day forecasting model."),
    }
    (report_dir / "xgboost_gold_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved full model:", model_dir / "xgboost_gold_pipeline.joblib")
    print("Saved metrics:", report_dir / "xgboost_gold_metrics.json")


def main():
    parser = argparse.ArgumentParser(description="XGBoost for Citi Bike station-day demand")
    parser.add_argument("--features", type=Path, default=Path("data/feature_store"),
                        help="Extracted feature_store folder or original feature_store.zip")
    parser.add_argument("--labels", type=Path, default=Path("data/label_store"),
                        help="Extracted label_store folder or original label_store.zip")
    parser.add_argument("--include-weather", action="store_true",
                        help="Retrospective only: observed same-day weather can leak future information")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    for p in [args.features, args.labels]:
        if not p.exists():
            parser.error(f"Not found: {p}. Confirm your working folder using pwd and ls data")
    df, cols = prepare_data(args.features, args.labels, args.include_weather)
    fit_model(df, cols, args.model_dir, args.report_dir, args.include_weather)


if __name__ == "__main__":
    main()
