import json
from pathlib import Path
from datetime import datetime

JOURNAL_DIR = Path("ml/v5_journal")
JOURNAL_FILE = JOURNAL_DIR / "journal.jsonl"
STATUS_FILE = JOURNAL_DIR / "status.json"

def init_journal():
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    if not STATUS_FILE.exists():
        STATUS_FILE.write_text(json.dumps({
            "prediction_count": 0,
            "evaluated_count": 0,
            "pending_count": 0,
            "by_symbol_precision": {},
            "by_regime_performance": {},
            "by_threshold_performance": {}
        }))

def append_journal(record):
    init_journal()
    with open(JOURNAL_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        
    # Update status summary
    status = json.loads(STATUS_FILE.read_text())
    status["prediction_count"] += 1
    status["pending_count"] += 1
    STATUS_FILE.write_text(json.dumps(status, indent=2))
