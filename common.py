from __future__ import annotations
import json, time, hashlib, os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_CONFIG = ROOT / "config.example.json"
USER_CONFIG = ROOT / "config.json"
load_dotenv(ROOT / ".env")

INTERVAL_MS = {
    "1": 60_000, "3": 180_000, "5": 300_000, "15": 900_000,
    "30": 1_800_000, "60": 3_600_000, "120": 7_200_000,
    "240": 14_400_000, "360": 21_600_000, "720": 43_200_000,
    "D": 86_400_000, "W": 604_800_000,
}

def load_config() -> dict[str, Any]:
    path = USER_CONFIG if USER_CONFIG.exists() else DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = ["symbols", "massive_tickers", "intervals", "history_days", "massive", "bybit"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    cfg["symbols"] = [str(x).upper() for x in cfg["symbols"]]
    cfg["intervals"] = [str(x) for x in cfg["intervals"]]
    return cfg

def massive_api_key(required: bool = True) -> str:
    key = os.getenv("MASSIVE_API_KEY", "").strip()
    if required and not key:
        raise RuntimeError("MASSIVE_API_KEY is missing. Copy .env.example to .env and add your key locally.")
    return key

def now_ms() -> int:
    return int(time.time() * 1000)

def utc_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def is_closed_candle(start_ms: int, interval: str, at_ms: int | None = None) -> bool:
    at_ms = now_ms() if at_ms is None else at_ms
    if interval not in INTERVAL_MS:
        return start_ms < at_ms
    return start_ms + INTERVAL_MS[interval] <= at_ms
