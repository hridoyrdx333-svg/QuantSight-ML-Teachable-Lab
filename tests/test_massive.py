import unittest
from collect_historical import massive_minute_df, resample_bars

class MassiveTransformTests(unittest.TestCase):
    def test_transform_and_resample(self):
        rows=[]
        base=1_700_000_000_000
        base -= base % 300_000
        for i in range(5):
            rows.append({"t":base+i*60_000,"o":100+i,"h":102+i,"l":99+i,"c":101+i,"v":10+i})
        df=massive_minute_df(rows,"BTCUSDT","X:BTCUSD")
        out=resample_bars(df,"5")
        self.assertEqual(len(out),1)
        self.assertEqual(float(out.iloc[0]["open"]),100.0)
        self.assertEqual(float(out.iloc[0]["close"]),105.0)
        self.assertEqual(float(out.iloc[0]["high"]),106.0)
        self.assertEqual(float(out.iloc[0]["low"]),99.0)
        self.assertEqual(float(out.iloc[0]["volume"]),60.0)

if __name__=="__main__":
    unittest.main()
