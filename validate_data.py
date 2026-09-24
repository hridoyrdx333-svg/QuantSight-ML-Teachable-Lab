from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from common import DATA_DIR, INTERVAL_MS, load_config, now_ms, is_closed_candle, file_sha256

def inspect_candles(path: Path, symbol: str, interval: str) -> dict:
    result = {
        "path": str(path.relative_to(DATA_DIR.parent)),
        "symbol": symbol,
        "interval": interval,
        "exists": path.exists(),
        "status": "FAIL",
        "issues": [],
    }
    if not path.exists():
        result["issues"].append("missing_file")
        return result

    df = pd.read_parquet(path)
    result["rows"] = int(len(df))
    result["sha256"] = file_sha256(path)
    needed = {"symbol","interval","start_ms","open","high","low","close"}
    missing = sorted(needed - set(df.columns))
    if missing:
        result["issues"].append(f"missing_columns:{missing}")
        return result
    if df.empty:
        result["issues"].append("empty")
        return result

    dup = int(df.duplicated(["symbol","interval","start_ms"]).sum())
    result["duplicates"] = dup
    if dup:
        result["issues"].append(f"duplicate_keys:{dup}")

    bad_ohlc = int(((df["high"] < df["low"]) |
                    (df["open"] < df["low"]) | (df["open"] > df["high"]) |
                    (df["close"] < df["low"]) | (df["close"] > df["high"])).sum())
    result["bad_ohlc"] = bad_ohlc
    if bad_ohlc:
        result["issues"].append(f"bad_ohlc:{bad_ohlc}")

    current_open = sum(not is_closed_candle(int(x), interval, now_ms()) for x in df["start_ms"])
    result["incomplete_candles"] = int(current_open)
    if current_open:
        result["issues"].append(f"incomplete_candles:{current_open}")

    if interval in INTERVAL_MS:
        s = pd.Series(sorted(df["start_ms"].astype("int64").unique()))
        gaps = []
        if len(s) > 1:
            diff = s.diff().dropna()
            exp = INTERVAL_MS[interval]
            for idx in diff[diff != exp].index[:100]:
                gaps.append({
                    "previous_ms": int(s.iloc[idx-1]),
                    "current_ms": int(s.iloc[idx]),
                    "difference_ms": int(diff.loc[idx]),
                    "expected_ms": exp,
                })
        result["gap_count_capped"] = len(gaps)
        result["gap_examples"] = gaps[:20]
        if gaps:
            result["issues"].append(f"missing_interval_gaps_detected:{len(gaps)}")

    result["start_ms"] = int(df["start_ms"].min())
    result["end_ms"] = int(df["start_ms"].max())
    result["status"] = "PASS" if not result["issues"] else "CHECK"
    return result

def inspect_funding(path: Path, symbol: str) -> dict:
    r = {"path": str(path.relative_to(DATA_DIR.parent)), "symbol": symbol, "exists": path.exists(), "status":"FAIL", "issues":[]}
    if not path.exists():
        r["issues"].append("missing_file")
        return r
    df = pd.read_parquet(path)
    r["rows"] = int(len(df))
    r["sha256"] = file_sha256(path)
    if df.empty:
        r["issues"].append("empty")
    elif "timestamp_ms" not in df.columns:
        r["issues"].append("missing_timestamp_ms")
    else:
        d = int(df.duplicated(["symbol","timestamp_ms"]).sum())
        r["duplicates"] = d
        if d: r["issues"].append(f"duplicate_keys:{d}")
        r["start_ms"] = int(df["timestamp_ms"].min())
        r["end_ms"] = int(df["timestamp_ms"].max())
    r["status"] = "PASS" if not r["issues"] else "CHECK"
    return r

def main():
    cfg = load_config()
    report = {
        "phase": 1,
        "generated_at_ms": now_ms(),
        "historical": [],
        "funding": [],
        "summary": {},
    }
    for symbol in cfg["symbols"]:
        for interval in cfg["intervals"]:
            p = DATA_DIR / "raw" / "trade_kline" / symbol / f"{interval}.parquet"
            report["historical"].append(inspect_candles(p, symbol, interval))
        fp = DATA_DIR / "raw" / "funding" / symbol / "funding.parquet"
        report["funding"].append(inspect_funding(fp, symbol))

    checks = report["historical"] + report["funding"]
    report["summary"] = {
        "pass": sum(x["status"]=="PASS" for x in checks),
        "check": sum(x["status"]=="CHECK" for x in checks),
        "fail": sum(x["status"]=="FAIL" for x in checks),
        "overall": "PASS" if all(x["status"]=="PASS" for x in checks) else "CHECK",
    }
    out = DATA_DIR / "validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("Report:", out)

if __name__ == "__main__":
    main()
