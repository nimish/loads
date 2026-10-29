# Reproducible county baseline comparison

This directory fixes the definitions of **calendar-and-weather regression (OLS)**
and **calendar-and-weather xgboost**, and compares them with TSGAM using the same
county inputs and training/scoring rows. See [MODELS.md](MODELS.md) for equations.

The OLS calendar is additive: 23 hour, 6 weekday and 11 month indicators, omitting
hour 0, Monday and January. Temperature and humidity enter at lags 0, 1 and 24 hours,
with training-only standardization and elementwise squares. There are no interaction
terms. XGBoost receives the same indicators and six standardized weather inputs,
without explicit squares; its trees can learn interactions.

## Install the frozen environment

From this directory, with `uv` installed:

```sh
uv sync --project environment --locked
uv run --project environment --locked python environment/verify_environment.py
uv run --project environment --locked python -m unittest discover -s tests -v
uv run --project environment --locked python example.py --holdout 5d
```

The example uses synthetic data and does not reproduce county results. The lock
pins Python package versions and all eight Git dependencies to commits, including
the underlying `loads` implementation. It intentionally installs that pinned
implementation, not an editable copy of the surrounding checkout.

The environment was reconstructed after the experiments. It preserves the original
missing Plotly dependency and registry-installed pvlib through explicit overrides.
Read [environment/README.txt](environment/README.txt) for the provenance boundary,
platform limits and clean-install verification. Python 3.12.6/macOS arm64 was tested.

## Input data

Obtain the frozen county input bundle from the experiment owner, and extract its
`inputs/` directory here (or pass another directory with `--inputs`). The bundle
contains 407 gzip CSV files. [inputs.json](inputs.json) records their exact filenames
and SHA-256 hashes. The runner checks every selected file before fitting; it never
downloads or rebuilds inputs from changing external services.

Each file has a timestamp index and `temperature_degF`, `humidity_pc`, and `actual`
(MW) columns on the complete 2018 UTC hourly grid. The data files are distributed
separately from Git; this repository alone is sufficient for the synthetic example,
but the exact county comparison requires the input bundle. Recompressing or modifying
a CSV changes its checksum and is rejected.

The supplied archive is named `county-baseline-inputs.zip`; its checksum and size
are recorded in [reference/input_bundle.json](reference/input_bundle.json).

## Reproduce the original comparison

```sh
uv run --project environment --locked python compare.py \
  --inputs inputs --output outputs/historical \
  --tsgam-reg-weight 6e-5 --tsgam-diff-reg-weight 0.5
```

This runs 407 counties × 2 holdouts × 3 models = 2,442 fits, using eight workers
and one numerical thread per worker. It checks scores against the committed
[historical reference](reference/historical_metrics.csv). The standalone OLS uses
scikit-learn instead of the historical explicit-intercept SciPy solve, so equivalence
is numerical rather than bitwise. Score checks use `rtol=1e-8, atol=1e-8`.

For a quick real-data reproduction check, append:

```sh
--counties "Los Angeles CA" "Maricopa AZ" "Spokane WA"
```

## Current fixed regularization

The new experiment setting holds MultiPeriodic and both weather-spline `reg_weight`
values at `1e-4`. The difference penalty defaults to `0.5`; the sweep did not select
a new permanent difference penalty. To run this comparison:

```sh
uv run --project environment --locked python compare.py \
  --inputs inputs --output outputs/reg-1e-4
```

Use `--tsgam-diff-reg-weight 0.1` (and a fresh output directory) to evaluate another
difference penalty, or `--models ols xgboost` to run just the conventional baselines.
These options change the experiment configuration, not package defaults. Historical
OLS/XGBoost scores are checked for every setting; historical TSGAM scores are checked
only when both original weights are selected.

## Outputs and verification

Each run records its configuration, code/lock/input hashes, per-fit metrics, prediction
files and hashes, training/scoring timestamp hashes, warnings and elapsed time.
`summary.csv` reports unweighted means across counties. Interrupted runs resume by
repeating the same command. A changed model or source manifest requires a new output
directory. Append `--verify-only` to repeat verification without fitting.

The first 24 hours are excluded from all models' training. The original 5d and 5w
scoring masks are preserved: `day_of_year % 5 == 4` and
`(day_of_year // 7) % 5 == 4`, respectively. These are recurring reconstruction
holdouts, not forecast horizons. Weather is observed and the targets are modeled
county loads. Regularization sweeps reuse these holdouts and constitute tuning,
not independent validation.

The directory retains earlier validation records as historical evidence. The branch's
fresh smoke-test results are in `reference/branch_validation.json`; they do not claim
that the complete 2,442-fit study was rerun during branch preparation.
