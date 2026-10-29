import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from calendar_weather import make_features, make_ols, make_xgboost, WEATHER_FEATURES
from example import synthetic_data, holdout_mask


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_data()
        cls.X = make_features(cls.frame)
        cls.y = cls.frame.actual.reindex(cls.X.index)

    def test_reference_categories_and_dimensions(self):
        for factory, columns in [(make_ols, 52), (make_xgboost, 46)]:
            model = factory()
            design = model.named_steps["features"].fit_transform(self.X)
            self.assertEqual(design.shape, (8736, columns))
            reference = self.X.loc[["2018-01-08T00:00:00Z"]]
            encoded = model.named_steps["features"].transform(reference)
            np.testing.assert_array_equal(encoded[0, :40], 0)
            names = model.named_steps["features"].get_feature_names_out()
            self.assertEqual(len(names), columns)
            if factory is make_ols:
                np.testing.assert_array_equal(design[:, -6:], design[:, 40:46] ** 2)

    def test_hourly_lags_and_holdout_counts(self):
        i = pd.Timestamp("2018-01-04T01:00:00Z")
        for column in ("temperature_degF", "humidity_pc"):
            for lag in (0, 1, 24):
                self.assertEqual(self.X.loc[i, f"{column}_lag{lag}"],
                                 self.frame.loc[i - pd.Timedelta(hours=lag), column])
        self.assertEqual(holdout_mask(self.X.index, "5d").sum(), 1752)
        self.assertEqual(holdout_mask(self.X.index, "5w").sum(), 1680)

    def test_preprocessing_is_training_only(self):
        train = ~holdout_mask(self.X.index, "5d")
        model = make_ols().fit(self.X[train], self.y[train])
        scaler = model.named_steps["features"].named_transformers_["weather"].named_steps["standardize"]
        np.testing.assert_allclose(scaler.mean_, self.X.loc[train, WEATHER_FEATURES].mean())
        before = scaler.mean_.copy()
        changed = self.X.loc[~train].copy()
        changed[WEATHER_FEATURES] += 10000
        model.predict(changed)
        np.testing.assert_array_equal(scaler.mean_, before)

    def test_exact_additive_signal_and_unseen_prediction_window(self):
        train = ~holdout_mask(self.X.index, "5w")
        model = make_ols()
        design = model.named_steps["features"].fit_transform(self.X.loc[train])
        slopes = np.random.default_rng(7).normal(size=52)
        all_design = model.named_steps["features"].transform(self.X)
        y = 100 + all_design @ slopes
        model.fit(self.X.loc[train], y[train])
        np.testing.assert_allclose(model.predict(self.X.loc[~train]), y[~train], atol=1e-10)
        # Predicting a supplied interval plus its 24-hour history equals selecting
        # that interval from features built on the complete original weather grid.
        window = self.frame.loc["2018-05-01":"2018-05-03"]
        X_window = make_features(window)
        np.testing.assert_allclose(model.predict(X_window), model.predict(self.X.loc[X_window.index]))

    def test_both_models_ignore_holdout_targets(self):
        train = ~holdout_mask(self.X.index, "5d")
        changed = self.y.copy()
        changed.loc[~train] = 1e12
        for factory in (make_ols, make_xgboost):
            original = factory().fit(self.X.loc[train], self.y.loc[train])
            second = clone(original).fit(self.X.loc[train], changed.loc[train])
            np.testing.assert_array_equal(original.predict(self.X.loc[~train]),
                                          second.predict(self.X.loc[~train]))

    def test_invalid_timeline_rejected_and_timezone_normalized(self):
        for invalid in [self.frame.iloc[::-1], self.frame.drop(self.frame.index[48]),
                        pd.concat([self.frame.iloc[:2], self.frame])]:
            with self.assertRaises(ValueError):
                make_features(invalid)
        naive = self.frame.copy()
        naive.index = naive.index.tz_localize(None)
        with self.assertRaises(ValueError):
            make_features(naive)
        pd.testing.assert_frame_equal(make_features(self.frame.tz_convert("America/Los_Angeles")), self.X)


if __name__ == "__main__":
    unittest.main()
