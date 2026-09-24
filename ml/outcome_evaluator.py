import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bybit_rest import BybitPublicREST

JOURNAL_DIR = Path("ml/v5_journal")
JOURNAL_FILE = JOURNAL_DIR / "journal.jsonl"
OUTCOMES_FILE = JOURNAL_DIR / "outcomes.jsonl"
STATUS_FILE = JOURNAL_DIR / "status.json"

def fetch_close_price(client, symbol, timestamp_ms):
    klines = client.trade_klines(symbol, "5", timestamp_ms, timestamp_ms + 5*60*1000)
    for k in klines:
        if int(k[0]) == timestamp_ms:
            return float(k[4])
    return None

def main():
    if not JOURNAL_FILE.exists():
        print("No journal found.")
        return

    client = BybitPublicREST("https://api.bytick.com")
    now_ms = int(time.time() * 1000)

    predictions = []
    with open(JOURNAL_FILE, "r") as f:
        for line in f:
            if not line.strip(): continue
            predictions.append(json.loads(line))
            
    outcomes_by_key = {}
    if OUTCOMES_FILE.exists():
        with open(OUTCOMES_FILE, "r") as f:
            for line in f:
                if not line.strip(): continue
                o = json.loads(line)
                key = f"{o['feature_timestamp']}_{o['symbol']}"
                outcomes_by_key[key] = o

    changed = False
    new_outcomes = []
    for p in predictions:
        key = f"{p['feature_timestamp']}_{p['symbol']}"
        if key not in outcomes_by_key:
            start_ms = p["feature_timestamp"]
            target_ms = start_ms + 30 * 60 * 1000
            
            if now_ms >= target_ms + 5 * 60 * 1000:
                print(f"Evaluating {p['symbol']} at {start_ms}...")
                px_start = fetch_close_price(client, p["symbol"], start_ms)
                px_end = fetch_close_price(client, p["symbol"], target_ms)
                
                if px_start is not None and px_end is not None:
                    ret = (px_end - px_start) / px_start
                    o = {
                        "feature_timestamp": start_ms,
                        "symbol": p["symbol"],
                        "eval_time": utc_now_iso() if 'utc_now_iso' in globals() else str(now_ms),
                        "px_start": px_start,
                        "px_end": px_end,
                        "future_return": ret,
                        "future_30m_outcome": 1 if ret > 0 else 0,
                    }
                    outcomes_by_key[key] = o
                    new_outcomes.append(o)
                    changed = True
                else:
                    print(f"Missing price data for {p['symbol']} {start_ms} -> {target_ms}")

    if new_outcomes:
        with open(OUTCOMES_FILE, "a") as f:
            for o in new_outcomes:
                f.write(json.dumps(o) + "\n")
                
    evaluated = len(outcomes_by_key)
    pending = len(predictions) - evaluated
    
    by_symbol_pass = {}
    by_regime_pass = {}
    by_thresh_pass = {}
    
    by_symbol_reject = {}
    
    for p in predictions:
        key = f"{p['feature_timestamp']}_{p['symbol']}"
        if key not in outcomes_by_key:
            continue
            
        o = outcomes_by_key[key]
        sym = p["symbol"]
        reg = str(p.get("market_regime", "unknown"))
        thr = str(p["threshold"])
        decision = p["decision"]
        
        outcome_up = o["future_30m_outcome"] == 1
        
        def add_metric(d, k, is_correct):
            if k not in d: d[k] = {"evaluated": 0, "correct": 0}
            d[k]["evaluated"] += 1
            if is_correct: d[k]["correct"] += 1

        if decision == "PASS":
            add_metric(by_symbol_pass, sym, outcome_up)
            add_metric(by_regime_pass, reg, outcome_up)
            add_metric(by_thresh_pass, thr, outcome_up)
        elif decision == "REJECT":
            add_metric(by_symbol_reject, sym, not outcome_up)

    def calc_prec(d):
        for k, v in d.items():
            v["precision"] = v["correct"] / v["evaluated"] if v["evaluated"] > 0 else 0.0
            
    calc_prec(by_symbol_pass)
    calc_prec(by_regime_pass)
    calc_prec(by_thresh_pass)
    calc_prec(by_symbol_reject)
    
    status = {
        "prediction_count": len(predictions),
        "evaluated_count": evaluated,
        "pending_count": pending,
        "PASS_by_symbol_precision": by_symbol_pass,
        "PASS_by_regime_performance": by_regime_pass,
        "PASS_by_threshold_performance": by_thresh_pass,
        "REJECT_accuracy": by_symbol_reject
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
