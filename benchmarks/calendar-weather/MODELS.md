# Calendar-and-weather baseline models

Fit each model separately for each county. Let $y_t$ denote hourly load in MW,
$\mathcal I$ the training timestamps, and $T_t$ and $R_t$ temperature and relative
humidity. Calendar variables use **UTC**: $H_t\in\{0,\ldots,23\}$,
$D_t\in\{0,\ldots,6\}$ (Monday is 0), and $M_t\in\{1,\ldots,12\}$.

## Weather preprocessing

For $v\in\{T,R\}$ and lag $\ell\in L=\{0,1,24\}$ hours, define

$$
\mu_{v,\ell}=\frac{1}{|\mathcal I|}\sum_{t\in\mathcal I}v_{t-\ell},
\qquad
s_{v,\ell}=\sqrt{\frac{1}{|\mathcal I|}\sum_{t\in\mathcal I}
\left(v_{t-\ell}-\mu_{v,\ell}\right)^2},
\qquad
z_{v,\ell,t}=\frac{v_{t-\ell}-\mu_{v,\ell}}{s_{v,\ell}}.
$$

Means and scales are learned **only from training rows** and reused at prediction
time. Constant features use scale 1 (with scikit-learn's near-constant safeguard).
Construct lags on the complete hourly timeline before splitting; omit the first
24 hours. Neither model uses lagged load.

## Calendar-and-weather regression (OLS)

$$
\begin{aligned}
\widehat y_t={}&\beta_0
+\sum_{h=1}^{23}a_h\,\mathbf{1}\{H_t=h\}
+\sum_{d=1}^{6}b_d\,\mathbf{1}\{D_t=d\}
+\sum_{m=2}^{12}c_m\,\mathbf{1}\{M_t=m\}\\
&+\sum_{v\in\{T,R\}}\sum_{\ell\in L}
\left(\alpha_{v,\ell}z_{v,\ell,t}
+\delta_{v,\ell}z_{v,\ell,t}^{2}\right),\\[3pt]
\widehat{\theta}={}&\underset{\theta}{\operatorname{arg\,min}}
\sum_{t\in\mathcal I}\left(y_t-\widehat y_t(\theta)\right)^2.
\end{aligned}
$$

Hour 0, Monday, and January are reference categories. There are **53 coefficients**:
one intercept, 40 calendar indicators, six linear weather terms, and six squares.
Both temperature and humidity, including their lags, enter linearly **and**
quadratically; squares are formed after standardization. Calendar effects are
additive. There are no interactions, regularization, target transformations, or
prediction clipping. Implemented by `make_ols()`.

## Calendar-and-weather xgboost

Let $\mathbf C_t\in\mathbb R^{40}$ contain the same calendar indicators. The input is

$$
\mathbf u_t=
\left[\mathbf C_t^\top,
z_{T,0,t},z_{T,1,t},z_{T,24,t},
z_{R,0,t},z_{R,1,t},z_{R,24,t}\right]^\top\in\mathbb R^{46}.
$$

$$
\widehat y_t=b+0.05\sum_{k=1}^{500}f_k(\mathbf u_t),
\qquad \operatorname{depth}(f_k)\leq 4.
$$

Here $b$ is XGBoost's training-derived initial prediction and $f_k$ are sequentially
fitted regression trees. Use squared-error loss, `min_child_weight=10`,
`subsample=0.8`, `colsample_bytree=1.0`, `reg_lambda=1.0`, `reg_alpha=0.0`,
`gamma=0.0`, `tree_method="hist"`, `max_bin=256`, `random_state=42`, and `n_jobs=1`.
There are no explicit squared features, early stopping, or holdout tuning. Trees
can learn nonlinearities and interactions implicitly. Implemented by `make_xgboost()`.

## Matched comparison

Use identical training and scoring timestamps for both baselines and TSGAM.
The supplied example reproduces the study's interspersed holdout masks, with
one-based day of year $q_t$:

$$
\mathcal H_{5d}=\{t:q_t\bmod5=4\},\qquad
\mathcal H_{5w}=\{t:\lfloor q_t/7\rfloor\bmod5=4\}.
$$

These are reconstruction holdouts, not 5-day or 5-week forecast horizons.
Using observed weather makes the comparison conditional on that weather.
See [README.md](README.md) for installation and runnable examples.
