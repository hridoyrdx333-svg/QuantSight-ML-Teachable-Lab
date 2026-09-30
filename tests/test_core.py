import unittest
from common import is_closed_candle, INTERVAL_MS
from collect_live import LocalOrderBook

class CoreTests(unittest.TestCase):
    def test_closed_candle(self):
        start = 1_000_000
        self.assertTrue(is_closed_candle(start, "5", start + INTERVAL_MS["5"]))
        self.assertFalse(is_closed_candle(start, "5", start + INTERVAL_MS["5"] - 1))

    def test_orderbook_snapshot_delta(self):
        b = LocalOrderBook()
        b.apply({
            "b":[["100","2"],["99","3"]],
            "a":[["101","4"],["102","5"]],
            "seq":1,
            "cts":1000,
        }, "snapshot", 1001)
        f = b.feature_row("BTCUSDT", 50, 1001)
        self.assertAlmostEqual(f["best_bid"], 100)
        self.assertAlmostEqual(f["best_ask"], 101)
        b.apply({
            "b":[["100","0"],["100.5","1"]],
            "a":[["101","2"]],
            "seq":2,
            "cts":1100,
        }, "delta", 1101)
        f = b.feature_row("BTCUSDT", 50, 1101)
        self.assertAlmostEqual(f["best_bid"], 100.5)
        self.assertAlmostEqual(f["ask_depth"], 7.0)

if __name__ == "__main__":
    unittest.main()
