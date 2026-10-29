"""Calendar-and-weather reference models built from standard sklearn pipelines.

Build lagged features on the complete weather timeline BEFORE splitting rows.
Fit preprocessing only through pipeline.fit(X_train, y_train).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

CALENDAR = ["hour", "weekday", "month"]
WEATHER = ["temperature_degF", "humidity_pc"]
LAGS = (0, 1, 24)
WEATHER_FEATURES = [f"{name}_lag{lag}" for name in WEATHER for lag in LAGS]


def make_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Return 3 calendar columns and 6 raw weather-lag columns, indexed in UTC.

    ``weather`` must have a sorted, unique, timezone-aware DatetimeIndex with
    exactly hourly spacing, and finite temperature_degF and humidity_pc columns.
    Other columns are ignored. Lags refer to elapsed hours, not retained rows.
    The first 24 rows are omitted because they lack lag history. To predict a
    new interval, pass its weather plus the preceding 24 hours of weather.
    Missing hours or weather values raise ValueError; nothing is interpolated.
    No targets or fitted statistics are used here.
    """
    if not isinstance(weather.index, pd.DatetimeIndex) or weather.index.tz is None:
        raise ValueError("Use a timezone-aware DatetimeIndex; explicitly localize naive timestamps first.")
    index = weather.index.tz_convert("UTC")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Weather timestamps must be valid, sorted, and unique.")
    if len(index) <= max(LAGS):
        raise ValueError("Supply more than 24 hours, including lag history.")
    if not (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("Weather must form a complete hourly timeline before feature construction.")
    missing = set(WEATHER) - set(weather.columns)
    if missing:
        raise ValueError(f"Missing weather columns: {sorted(missing)}")
    values = weather[WEATHER].astype(float).copy()
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Weather values must all be finite.")
    values.index = index
    features = pd.DataFrame({"hour": index.hour, "weekday": index.day_of_week,
                             "month": index.month}, index=index)
    for name in WEATHER:
        for lag in LAGS:
            features[f"{name}_lag{lag}"] = values[name].shift(lag)
    return features.iloc[max(LAGS):].copy()


def _linear_and_squared(z):
    # Deliberately NOT PolynomialFeatures(degree=2): that would add interactions.
    return np.concatenate([z, z ** 2], axis=1)


def _squared_feature_names(transformer, names):
    return np.concatenate([names, [f"{name}^2" for name in names]])


def _preprocessor(*, squares: bool) -> ColumnTransformer:
    calendar = OneHotEncoder(
        categories=[list(range(24)), list(range(7)), list(range(1, 13))],
        drop=[0, 0, 1],  # Hour 0, Monday, January.
        handle_unknown="error", sparse_output=False,
    )
    weather_steps = [("standardize", StandardScaler())]
    if squares:
        weather_steps.append(("linear_and_squared", FunctionTransformer(
            _linear_and_squared, feature_names_out=_squared_feature_names,
        )))
    return ColumnTransformer([
        ("calendar", calendar, CALENDAR),
        ("weather", Pipeline(weather_steps), WEATHER_FEATURES),
    ], remainder="drop", sparse_threshold=0)


def make_ols() -> Pipeline:
    """Calendar-and-weather regression (OLS): 52 features plus an intercept.

    Additive calendar indicators and linear/quadratic standardized weather.
    No interaction terms, penalties, log transform, or prediction clipping.
    StandardScaler learns population moments on the supplied training rows.
    ``model.named_steps['regressor'].coef_`` contains 52 slopes;
    ``intercept_`` is the overall level. ``rank_`` diagnoses the centered design.
    """
    return Pipeline([
        ("features", _preprocessor(squares=True)),
        ("regressor", LinearRegression(fit_intercept=True)),
    ])


def make_xgboost() -> Pipeline:
    """Calendar-and-weather xgboost: 40 indicators and 6 standardized lags.

    Fixed settings, squared-error loss in original target units. The trees can
    learn interactions; no explicit polynomial features or early stopping.
    """
    return Pipeline([
        ("features", _preprocessor(squares=False)),
        ("regressor", XGBRegressor(
            objective="reg:squarederror", booster="gbtree", n_estimators=500,
            max_depth=4, learning_rate=0.05, min_child_weight=10,
            subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
            reg_alpha=0.0, gamma=0.0, tree_method="hist", max_bin=256,
            random_state=42, n_jobs=1,
        )),
    ])
