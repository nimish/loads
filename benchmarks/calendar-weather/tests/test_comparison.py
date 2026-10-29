import tempfile
from pathlib import Path
import unittest
import json
import copy
from unittest.mock import patch

import numpy as np
from compare import masks, read_input, cluster_inputs, tsgam_config, HERE
from example import synthetic_data


class ComparisonTests(unittest.TestCase):
    def test_cluster_transfer_covers_every_county(self):
        inputs = json.loads((HERE / "inputs.json").read_text())
        spec = json.loads((HERE / "clusters.json").read_text())
        enriched = cluster_inputs(inputs, spec)
        self.assertEqual(len(enriched), 407)
        expected = {"Lincoln OR": [4, 2, 3], "Spokane WA": [4, 5, 7],
                    "Multnomah OR": [2, 5, 5], "San Luis Obispo CA": [4, 5, 8],
                    "Maricopa AZ": [4, 3, 7], "Solano CA": [5, 2, 7]}
        for item in enriched:
            self.assertEqual(item["harmonics"], expected[item["medoid"]])
        member = next(item for item in enriched if item["county"] == "Ada ID")
        self.assertEqual(member["medoid"], "Multnomah OR")
        self.assertEqual(member["harmonics"], [2, 5, 5])
        with self.assertRaises(ValueError):
            cluster_inputs(inputs[:-1], spec)
        broken = copy.deepcopy(spec)
        broken["assignments"]["Ada ID"] = "Unknown"
        with self.assertRaises(ValueError):
            cluster_inputs(inputs, broken)

    def test_tsgam_configuration_is_explicit_and_isolated(self):
        from loads.estimator import Estimator, EstimatorConfig
        original = Estimator._estimator_config(EstimatorConfig())
        before = repr(original)
        for harmonics in ([4, 2, 3], [4, 5, 7], [2, 5, 5], [4, 5, 8], [4, 3, 7], [5, 2, 7]):
            config = tsgam_config(harmonics, 1e-4, 10.0, True)
            self.assertEqual(config.multi_periodic_config.num_harmonics, harmonics)
            self.assertEqual(config.multi_periodic_config.reg_weight, 1e-4)
            for new, old in zip(config.exog_config, original.exog_config):
                self.assertEqual(new.reg_weight, 1e-4)
                self.assertEqual(new.diff_reg_weight, 10.0)
                self.assertIs(new.whiten, True)
                self.assertEqual(new.lags, old.lags)
                self.assertEqual(new.n_knots, old.n_knots)
            self.assertEqual(config.outlier_config, original.outlier_config)
            self.assertEqual(config.solver_config, original.solver_config)
        self.assertEqual(repr(Estimator._estimator_config(EstimatorConfig())), before)

    def test_common_training_support(self):
        frame = synthetic_data()
        for holdout, counts in [("5d", (6984, 1752)), ("5w", (7056, 1680))]:
            train, test = masks(frame.index, holdout)
            self.assertEqual((train.sum(), test.sum()), counts)
            self.assertFalse(train[:24].any())
            self.assertFalse(test[:24].any())
            self.assertFalse((train & test).any())
            np.testing.assert_array_equal((train | test)[24:], True)

    def test_corrupt_input_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "county.csv.gz"
            path.write_bytes(b"not the frozen input")
            with patch("compare.pd.read_csv") as reader:
                with self.assertRaisesRegex(ValueError, "Input hash mismatch"):
                    read_input(path, "0" * 64)
                reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
