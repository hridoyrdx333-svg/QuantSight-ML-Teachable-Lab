import subprocess
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"

RUNTIME_DIR = ROOT / "ui" / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

def kill_lock(lock_file):
    lock_path = RUNTIME_DIR / lock_file
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip())
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        except Exception:
            pass
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

def start_proc(script, lock_file):
    lock_path = RUNTIME_DIR / lock_file
    kill_lock(lock_file)
    
    script_path = str(ROOT / script)
    
    p = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), script_path],
        creationflags=subprocess.CREATE_NO_WINDOW,
        cwd=str(ROOT),
    )
    lock_path.write_text(str(p.pid))

if __name__ == "__main__":
    start_proc("ui/app.py", "frontend.lock")
    start_proc("collect_live.py", "live_collector.lock")
    start_proc("ml/live_feature_builder.py", "builder.lock")
    start_proc("ml/shadow_runner.py", "shadow.lock")
    print('All processes started.')
    import time
    while True: time.sleep(1)
