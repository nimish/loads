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
  --profile historical
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

## Planned cluster-harmonic comparison (default profile)

Each county receives its assigned medoid's **harmonic counts**, then is fitted
independently on its own load/weather observations. Coefficients are not transferred.
The frozen [clusters.json](clusters.json) assignments match the original comparison's
407 counties exactly. The runner rejects incomplete coverage or unknown medoids.

| Medoid | Annual, weekly, daily harmonics |
|---|---|
| Lincoln OR | (4, 2, 3) |
| Spokane WA | (4, 5, 7) |
| Multnomah OR | (2, 5, 5) |
| San Luis Obispo CA | (4, 5, 8) |
| Maricopa AZ | (4, 3, 7) |
| Solano CA | (5, 2, 7) |

TSGAM uses `reg_weight=1e-4` for MultiPeriodic and both weather-spline blocks,
`diff_reg_weight=10` and `whiten=True` for both weather-spline blocks. Temperature
retains 10 knots and offsets [-3,-2,-1,0]; humidity retains 8 knots and offsets
[-2,-1,0]. The log-load target, CLARABEL solver and daily outlier component
(weight 1e-4) are unchanged. No trend, AR or interaction blocks are added.
OLS and XGBoost remain unchanged.

The reviewed parameters, cluster sizes, expected fit count and source hashes are
recorded in [reference/planned_cluster_comparison.json](reference/planned_cluster_comparison.json).

Prepare and inspect all 2,442 cases **without fitting**:

```sh
uv run --project environment --locked python compare.py \
  --inputs inputs --output outputs/cluster-harmonics --prepare-only
```

After reviewing `outputs/cluster-harmonics/manifest.json`, launch the same plan:

```sh
uv run --project environment --locked python compare.py \
  --inputs inputs --output outputs/cluster-harmonics
```

Use `--counties` for a smoke check or `--models ols xgboost` for just the conventional
baselines. Overrides `--tsgam-reg-weight`, `--tsgam-diff-reg-weight` and
`--whiten`/`--no-whiten` require a fresh output directory if they change the plan.
These options do not change package defaults. Historical OLS/XGBoost scores are
checked in every run. Historical TSGAM scores are checked only under the historical
profile with both original weights and whitening disabled.

## Outputs and verification

Each run records its configuration, code/lock/input hashes, per-fit metrics, prediction
files and hashes, training/scoring timestamp hashes, warnings and elapsed time.
`summary.csv` reports unweighted means across counties; `cluster_summary.csv`
groups the same metrics by assigned medoid. `paired_metrics.csv` and `comparisons.csv`
give county-level comparisons and aggregate TSGAM-versus-baseline differences/wins.
Interrupted runs resume by
repeating the same command. A changed model or source manifest requires a new output
directory. Append `--verify-only` to repeat verification without fitting.

The first 24 hours are excluded from all models' training. The original 5d and 5w
scoring masks are preserved: `day_of_year % 5 == 4` and
`(day_of_year // 7) % 5 == 4`, respectively. These are recurring reconstruction
holdouts, not forecast horizons. Weather is observed and the targets are modeled
county loads. Regularization sweeps reuse these holdouts and constitute tuning,
not independent validation.

The directory retains earlier validation records as historical evidence. The branch's
initial smoke-test results are in `reference/branch_validation.json` (the initial
runner revision). Cluster-profile checks are in `reference/cluster_validation.json`.
Neither claims that the complete 2,442-fit study was rerun during branch preparation.
