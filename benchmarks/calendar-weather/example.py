"""Run with synthetic data or an hourly CSV; see README.md for the contract."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_weather import make_features, make_ols, make_xgboost


def synthetic_data():
    """Self-contained example only; this is not the county benchmark dataset."""
    index = pd.date_range("2018-01-01", periods=8760, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    t = np.arange(len(index))
    temperature = 55 + 20 * np.sin(2 * np.pi * t / 8760) + rng.normal(0, 4, len(t))
    humidity = 55 - 10 * np.sin(2 * np.pi * t / 8760) + rng.normal(0, 5, len(t))
    load = (100 + 10 * (index.hour >= 8) + 7 * (index.day_of_week < 5)
            + .04 * (temperature - 60) ** 2 + .1 * humidity + rng.normal(0, 2, len(t)))
    return pd.DataFrame({"temperature_degF": temperature, "humidity_pc": humidity,
                         "actual": load}, index=index)


def holdout_mask(index, scheme):
    """The original recurring county reconstruction masks, not forecast horizons."""
    day = index.day_of_year
    return np.asarray((day if scheme == "5d" else day // 7) % 5 == 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="First column is a timestamp with explicit timezone/offset.")
    parser.add_argument("--holdout", choices=["5d", "5w"], default="5d")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    if args.csv:
        frame = pd.read_csv(args.csv, index_col=0)
        # Reject naive timestamps rather than silently interpreting local time as UTC.
        stamps = [pd.Timestamp(value) for value in frame.index]
        if any(stamp.tzinfo is None for stamp in stamps):
            raise ValueError("CSV timestamps must include a timezone or UTC offset.")
        frame.index = pd.to_datetime(frame.index, utc=True)
    else:
        frame = synthetic_data()
    X = make_features(frame)
    y = frame.actual.reindex(X.index).astype(float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("This MAPE example requires finite, positive actual load.")
    test = holdout_mask(X.index, args.holdout)
    if not test.any() or test.all():
        raise ValueError("Data must contain both training and holdout observations.")
    result = pd.DataFrame({"actual": y[test]})
    print("Synthetic illustration" if args.csv is None else f"Data: {args.csv}")
    print(f"Training hours: {(~test).sum()}; holdout hours: {test.sum()}")
    for name, model in [("calendar-and-weather regression (OLS)", make_ols()),
                        ("calendar-and-weather xgboost", make_xgboost())]:
        model.fit(X.loc[~test], y.loc[~test])
        prediction = model.predict(X.loc[test])
        error = prediction - y.loc[test].to_numpy()
        result[name] = prediction
        print(json.dumps({"model": name, "mae_mw": float(np.abs(error).mean()),
                          "rmse_mw": float(np.sqrt(np.mean(error ** 2))),
                          "mape_pct": float(100 * np.mean(np.abs(error / y.loc[test])))}))
    result.to_csv(args.output, index_label="timestamp")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
