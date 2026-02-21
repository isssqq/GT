import math
import unittest

from src.vwap_auto_anchored_zscore import compute_vwap_auto_anchored_zscore


class TestVwapAutoAnchoredZScore(unittest.TestCase):
    def test_output_has_valid_values_after_anchor(self):
        high = [10, 11, 13, 12, 11, 12, 13, 14, 13, 12, 11, 10, 11, 12]
        low = [9, 10, 11, 10, 9, 10, 11, 12, 11, 10, 9, 8, 9, 10]
        close = [9.5, 10.5, 12.5, 11, 10, 11, 12.5, 13.5, 12, 11, 10, 9, 10, 11]
        volume = [100, 120, 150, 130, 110, 115, 160, 180, 140, 135, 120, 110, 105, 100]

        result = compute_vwap_auto_anchored_zscore(high, low, close, volume, left=2, right=2, mode="both")

        self.assertTrue(0 <= result.anchor_index < len(close))
        for i in range(result.anchor_index, len(close)):
            self.assertFalse(math.isnan(result.vwap[i]))
            self.assertFalse(math.isnan(result.zscore[i]))

    def test_invalid_input_length(self):
        with self.assertRaises(ValueError):
            compute_vwap_auto_anchored_zscore([1], [1], [1], [1, 2])


if __name__ == "__main__":
    unittest.main()
