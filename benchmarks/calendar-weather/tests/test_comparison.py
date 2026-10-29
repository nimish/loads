import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from compare import masks, read_input
from example import synthetic_data


class ComparisonTests(unittest.TestCase):
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
