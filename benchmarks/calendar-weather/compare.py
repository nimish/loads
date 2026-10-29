"""Reproduce matched county comparisons from hash-verified frozen input CSVs."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from calendar_weather import make_features, make_ols, make_xgboost
from example import holdout_mask

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_input(path, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError(f"Input hash mismatch: {path}")
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True)
    expected = pd.date_range("2018-01-01", "2019-01-01", freq="h", inclusive="left", tz="UTC")
    if not frame.index.equals(expected):
        raise ValueError("Expected the complete 2018 UTC hourly grid")
    frame = frame[["temperature_degF", "humidity_pc", "actual"]].astype(float)
    if not np.isfinite(frame.to_numpy()).all() or (frame.actual <= 0).any():
        raise ValueError("Expected finite weather and positive loads")
    return frame


def masks(index, holdout):
    if holdout not in ("5d", "5w"):
        raise ValueError(holdout)
    test = holdout_mask(index, holdout)
    valid = np.arange(len(index)) >= 24
    if not valid[test].all():
        raise ValueError("Warm-up would alter scoring support")
    return valid & ~test, test


def scores(actual, prediction):
    error = prediction - actual
    return {"mae_mw": float(np.abs(error).mean()),
            "rmse_mw": float(np.sqrt(np.mean(error ** 2))),
            "mape_pct": float(100 * np.mean(np.abs(error / actual)))}


def fit(case, inputs, output, reg_weight, diff_reg_weight):
    started = time.perf_counter()
    frame = read_input(inputs / case["file"], case["sha256"])
    training, test = masks(frame.index, case["holdout"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if case["family"] == "tsgam":
            from loads.estimator import Estimator, EstimatorConfig
            config = EstimatorConfig()
            original = Estimator._estimator_config(config)
            modified = replace(original,
                multi_periodic_config=replace(original.multi_periodic_config, reg_weight=reg_weight),
                exog_config=[replace(x, reg_weight=reg_weight, diff_reg_weight=diff_reg_weight)
                             for x in original.exog_config])
            model = Estimator._get_model(frame.loc[training, list(config.weather)],
                                        np.log(frame.loc[training, ["actual"]]), modified)
            prediction = np.exp(np.asarray(model(frame[list(config.weather)])).ravel())[test]
            configuration = repr(modified)
        else:
            X = make_features(frame)
            train_X, test_X = training[24:], test[24:]
            model = (make_ols if case["family"] == "ols" else make_xgboost)()
            model.fit(X.loc[train_X], frame.actual.iloc[24:].loc[train_X])
            prediction = model.predict(X.loc[test_X])
            configuration = repr(model)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    target = output / "predictions" / (case["id"] + ".npz")
    temporary = target.with_suffix(".tmp")
    actual = frame.actual.to_numpy()[test]
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, timestamp_ns=frame.index[test].as_unit("ns").asi8,
                            actual_mw=actual, prediction_mw=prediction)
    temporary.replace(target)
    return {**case, **scores(actual, prediction), "configuration": configuration,
            "n_train": int(training.sum()), "n_holdout": int(test.sum()),
            "train_sha256": hashlib.sha256(frame.index[training].as_unit("ns").asi8.tobytes()).hexdigest(),
            "holdout_sha256": hashlib.sha256(frame.index[test].as_unit("ns").asi8.tobytes()).hexdigest(),
            "prediction_sha256": digest(target), "seconds": time.perf_counter() - started,
            "warnings": sorted({str(w.message) for w in caught})}


def verify(output, spec, historical):
    reference = pd.read_csv(HERE / "reference/historical_metrics.csv")
    rows = []
    for case in spec["cases"]:
        row = json.loads((output / "metrics" / (case["id"] + ".json")).read_text())
        path = output / "predictions" / (case["id"] + ".npz")
        if digest(path) != row["prediction_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {path}")
        with np.load(path) as saved:
            recomputed = scores(saved["actual_mw"], saved["prediction_mw"])
            if not np.isfinite(saved["prediction_mw"]).all():
                raise ValueError("Nonfinite saved predictions")
            for metric, value in recomputed.items():
                np.testing.assert_allclose(row[metric], value, rtol=1e-12)
            np.testing.assert_equal(len(saved["timestamp_ns"]), row["n_holdout"])
            assert hashlib.sha256(saved["timestamp_ns"].tobytes()).hexdigest() == row["holdout_sha256"]
        old_family = "tsgam_defaults" if case["family"] == "tsgam" else case["family"]
        old = reference.loc[(reference.county == case["county"]) &
                            (reference.holdout == case["holdout"]) &
                            (reference.family == old_family)].iloc[0]
        for key in ("n_train", "n_holdout", "train_sha256", "holdout_sha256"):
            assert row[key] == old[key], (case["id"], key)
        if historical or case["family"] != "tsgam":
            for key in ("mae_mw", "rmse_mw", "mape_pct"):
                np.testing.assert_allclose(row[key], old[key], rtol=1e-8, atol=1e-8,
                                           err_msg=f"Historical mismatch: {case['id']} {key}")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output / "metrics.csv", index=False)
    summary = df.groupby(["family", "holdout"])[["mae_mw", "rmse_mw", "mape_pct"]].mean()
    summary.to_csv(output / "summary.csv")
    atomic_json(output / "verification.json", {"verified_fits": len(rows),
        "expected_fits": len(spec["cases"]), "historical_tsgam": historical,
        "warning_fits": int(df.warnings.map(bool).sum())})
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counties", nargs="+")
    parser.add_argument("--models", nargs="+", choices=["ols", "xgboost", "tsgam"],
                        default=["ols", "xgboost", "tsgam"])
    parser.add_argument("--holdouts", nargs="+", choices=["5d", "5w"], default=["5d", "5w"])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--tsgam-reg-weight", type=float, default=1e-4)
    parser.add_argument("--tsgam-diff-reg-weight", type=float, default=0.5)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or any(not np.isfinite(x) or x < 0 for x in
                           (args.tsgam_reg_weight, args.tsgam_diff_reg_weight)):
        parser.error("Jobs must be positive and weights finite and nonnegative")
    subprocess.run([sys.executable, str(HERE / "environment/verify_environment.py")], check=True)
    inputs = json.loads((HERE / "inputs.json").read_text())
    if args.counties:
        missing = set(args.counties) - {x["county"] for x in inputs}
        if missing:
            parser.error(f"Unknown counties: {sorted(missing)}")
        inputs = [x for x in inputs if x["county"] in args.counties]
    for item in inputs:
        read_input(args.inputs / item["file"], item["sha256"])
    cases = [{**item, "holdout": h, "family": f,
              "id": item["county"].replace(" ", "_") + "__" + h + "__" + f}
             for item in inputs for h in sorted(set(args.holdouts)) for f in sorted(set(args.models))]
    source_files = [*HERE.glob("*.py"), HERE / "inputs.json",
                    HERE / "reference/historical_metrics.csv",
                    HERE / "environment/uv.lock", HERE / "environment/pyproject.toml",
                    HERE / "environment/verify_environment.py", HERE / "environment/installed-environment.json"]
    spec = {"cases": cases, "reg_weight": args.tsgam_reg_weight,
            "diff_reg_weight": args.tsgam_diff_reg_weight,
            "sources": {str(p.relative_to(HERE)): digest(p) for p in source_files}}
    historical = args.tsgam_reg_weight == 6e-5 and args.tsgam_diff_reg_weight == 0.5
    output = args.output.resolve()
    if args.verify_only and not (output / "manifest.json").exists():
        parser.error("No existing run to verify")
    for folder in ("metrics", "predictions"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != spec:
            parser.error("Run manifest changed; choose a new output directory")
    else:
        atomic_json(manifest, spec)
    if not args.verify_only:
        pending = [c for c in cases if not (output / "metrics" / (c["id"] + ".json")).exists()]
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(fit, c, args.inputs.resolve(), output,
                                   args.tsgam_reg_weight, args.tsgam_diff_reg_weight): c for c in pending}
            for n, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                atomic_json(output / "metrics" / (case["id"] + ".json"), future.result())
                if n % 50 == 0 or n == len(pending):
                    print(f"Completed {n}/{len(pending)} pending fits", flush=True)
    verify(output, spec, historical)


if __name__ == "__main__":
    main()
