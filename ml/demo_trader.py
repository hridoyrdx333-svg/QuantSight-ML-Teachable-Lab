import json
import time
import os
import hmac
import hashlib
import requests
import math
from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

JOURNAL_DIR = Path("ml/v5_journal")
JOURNAL_FILE = JOURNAL_DIR / "journal.jsonl"
DEMO_STATE_FILE = JOURNAL_DIR / "demo_state.json"

PLACE_ORDERS = False
KILL_SWITCH_ACTIVE = False

# Risk Settings
MAX_POSITIONS_PER_SYMBOL = 1
FIXED_DEMO_RISK_USDT = 10.0
MAX_RISK_PER_TRADE = 15.0
DAILY_LOSS_CAP = 50.0
STALE_FEATURE_SECONDS = 900 # 15 minutes

def get_signature(api_secret, param_str):
    return hmac.new(bytes(api_secret, "utf-8"), param_str.encode("utf-8"), hashlib.sha256).hexdigest()

def private_get(endpoint, api_key, api_secret, params=""):
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    param_str = timestamp + api_key + recv_window + params
    signature = get_signature(api_secret, param_str)
    
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": signature
    }
    
    url = f"https://api-demo.bybit.com{endpoint}"
    if params:
        url += f"?{params}"
        
    res = requests.get(url, headers=headers)
    return res.json()

def public_get(endpoint, params=""):
    url = f"https://api-demo.bybit.com{endpoint}"
    if params:
        url += f"?{params}"
    res = requests.get(url)
    return res.json()

def main():
    print("=" * 60)
    print("QuantSight V5 Demo Trader (DRY RUN)")
    print(f"PLACE_ORDERS = {PLACE_ORDERS}")
    print("=" * 60)
    
    if KILL_SWITCH_ACTIVE:
        print("FAIL: KILL SWITCH IS ACTIVE.")
        sys.exit(1)
    
    load_dotenv()
    
    api_key = os.getenv("BYBIT_DEMO_API_KEY")
    api_secret = os.getenv("BYBIT_DEMO_API_SECRET")
    
    if not api_key or not api_secret:
        print("FAIL: BYBIT_DEMO_API_KEY or BYBIT_DEMO_API_SECRET is missing.")
        sys.exit(1)
        
    print("Real Demo API Keys found.")
    
    print("Calling api-demo.bybit.com wallet-balance...")
    acc_res = private_get("/v5/account/wallet-balance", api_key, api_secret, "accountType=UNIFIED")
    if acc_res.get("retCode") != 0:
        print(f"FAIL: Real demo private API auth failed: {acc_res}")
        sys.exit(1)
        
    print(f"Real demo private API auth PASS. Account data returned (retCode=0).")
    
    # Check Daily Loss Cap
    today_start = int(time.time() / 86400) * 86400 * 1000
    pnl_res = private_get("/v5/position/closed-pnl", api_key, api_secret, f"category=linear&limit=100")
    if pnl_res.get("retCode") == 0:
        daily_loss = 0.0
        for p in pnl_res["result"]["list"]:
            if int(p["createdTime"]) >= today_start:
                pnl = float(p.get("closedPnl", 0))
                if pnl < 0:
                    daily_loss += abs(pnl)
        if daily_loss >= DAILY_LOSS_CAP:
            print(f"FAIL: Daily loss cap reached: {daily_loss} >= {DAILY_LOSS_CAP}")
            sys.exit(0)
    
    # Load State
    state = {"processed_signals": {}}
    if DEMO_STATE_FILE.exists():
        loaded_state = json.loads(DEMO_STATE_FILE.read_text())
        # Migration from old list format to dict format if needed
        if "processed_signals" in loaded_state:
            ps = loaded_state["processed_signals"]
            if isinstance(ps, list):
                state["processed_signals"] = {sig_id: "dry_run_validated" for sig_id in ps}
            else:
                state["processed_signals"] = ps
        
    # Load Journal
    if not JOURNAL_FILE.exists():
        print("FAIL: No journal found.")
        sys.exit(1)
        
    signals = []
    with open(JOURNAL_FILE, "r") as f:
        for line in f:
            if not line.strip(): continue
            signals.append(json.loads(line))
            
    # Find latest eligible PASS signal
    now_ms = int(time.time() * 1000)
    selected_sig = None
    for sig in reversed(signals):
        sig_id = f"{sig['feature_timestamp']}_{sig['symbol']}"
        
        if sig["decision"] != "PASS":
            continue
            
        if sig_id in state["processed_signals"]:
            continue
            
        if float(sig["probability"]) < float(sig["threshold"]):
            continue
            
        age_seconds = (now_ms - int(sig["feature_timestamp"])) / 1000.0
        if age_seconds > STALE_FEATURE_SECONDS:
            continue
            
        selected_sig = sig
        break
        
    if not selected_sig:
        print("NO ELIGIBLE PASS SIGNAL")
        print(f"\nOrders sent = 0")
        sys.exit(0)
        
    sym = selected_sig["symbol"]
    sig_id = f"{selected_sig['feature_timestamp']}_{sym}"
    
    print(f"\nEligible PASS signal found:")
    print(f"  Symbol: {sym}")
    print(f"  Feature Timestamp: {selected_sig['feature_timestamp']}")
    print(f"  Probability: {selected_sig['probability']}")
    print(f"  Locked Threshold: {selected_sig['threshold']}")
    
    # DO NOT persist processed signal ID yet. Wait for API/math/risk to complete.
    
    # Check open positions
    pos_res = private_get("/v5/position/list", api_key, api_secret, f"category=linear&symbol={sym}")
    if pos_res.get("retCode") != 0:
        print(f"FAIL: Failed to fetch positions: {pos_res}")
        sys.exit(1)
        
    open_positions = [p for p in pos_res["result"]["list"] if float(p["size"]) > 0]
    if len(open_positions) >= MAX_POSITIONS_PER_SYMBOL:
        print(f"FAIL: Max position reached for {sym}.")
        sys.exit(0)
        
    # Fetch real instrument metadata
    inst_res = public_get("/v5/market/instruments-info", f"category=linear&symbol={sym}")
    if inst_res.get("retCode") != 0 or not inst_res["result"]["list"]:
        print(f"FAIL: Failed to fetch instrument metadata: {inst_res}")
        sys.exit(1)
        
    inst = inst_res["result"]["list"][0]
    min_qty = float(inst["lotSizeFilter"]["minOrderQty"])
    qty_step = float(inst["lotSizeFilter"]["qtyStep"])
    tick_size = float(inst["priceFilter"]["tickSize"])
    min_notional = float(inst.get("lotSizeFilter", {}).get("minNotionalValue", 0.0))
    
    print(f"\nInstrument Metadata:")
    print(f"  minOrderQty: {min_qty}")
    print(f"  qtyStep: {qty_step}")
    print(f"  tickSize: {tick_size}")
    print(f"  minNotional: {min_notional}")
    
    # Get current price
    tick_res = public_get("/v5/market/tickers", f"category=linear&symbol={sym}")
    if tick_res.get("retCode") != 0 or not tick_res["result"]["list"]:
        print("FAIL: Failed to fetch ticker.")
        sys.exit(1)
        
    current_price = float(tick_res["result"]["list"][0]["markPrice"])
    print(f"  markPrice: {current_price}")
    
    # Explicit deterministic risk rule
    sl_dist = 0.01
    tp_dist = 0.02
    
    sl_price = current_price * (1 - sl_dist)
    tp_price = current_price * (1 + tp_dist)
    
    def round_tick(val, tick):
        return round(val / tick) * tick
        
    sl_price = round_tick(sl_price, tick_size)
    tp_price = round_tick(tp_price, tick_size)
    
    # Calculate quantity
    price_risk = current_price - sl_price
    if price_risk <= 0:
        print("FAIL: Invalid price risk.")
        sys.exit(1)
        
    raw_qty = FIXED_DEMO_RISK_USDT / price_risk
    # FLOOR to qty step, NEVER round up
    qty = math.floor(raw_qty / qty_step) * qty_step
    
    # Ensure min order qty
    if qty < min_qty:
        print(f"FAIL: Calculated qty {qty} is less than minOrderQty {min_qty}. Order aborted.")
        sys.exit(0)
        
    # Ensure min notional
    notional = qty * current_price
    if min_notional > 0 and notional < min_notional:
        print(f"FAIL: Calculated notional {notional} is less than minNotional {min_notional}. Order aborted.")
        sys.exit(0)
        
    # Ensure risk max
    actual_risk = qty * price_risk
    if actual_risk > MAX_RISK_PER_TRADE:
        print(f"FAIL: Calculated risk {actual_risk} exceeds MAX_RISK_PER_TRADE {MAX_RISK_PER_TRADE}. Order aborted.")
        sys.exit(0)
    
    print(f"\nCalculations:")
    print(f"  calculated SL: {sl_price}")
    print(f"  calculated TP: {tp_price}")
    print(f"  calculated qty: {qty}")
    print(f"  calculated max risk: {actual_risk} USDT")
    
    order = {
        "symbol": sym,
        "side": "Buy",
        "orderType": "Market",
        "qty": str(qty),
        "takeProfit": str(tp_price),
        "stopLoss": str(sl_price),
        "timeInForce": "GTC",
        "reduceOnly": False
    }
    
    print(f"\nIntended Demo Order Payload:\n{json.dumps(order, indent=2)}")
    
    # Final step: Persist state now that everything has passed
    if PLACE_ORDERS:
        # FUTURE: Send actual order to Bybit via private_post(...)
        # if private_post(...)["retCode"] == 0:
        #    state["processed_signals"][sig_id] = "executed"
        #    print("Order sent successfully.")
        pass
    else:
        state["processed_signals"][sig_id] = "dry_run_validated"
        print(f"\nOrders sent = 0")
        print("DRY-RUN PASS")
        
    DEMO_STATE_FILE.write_text(json.dumps(state, indent=2))

if __name__ == "__main__":
    main()
