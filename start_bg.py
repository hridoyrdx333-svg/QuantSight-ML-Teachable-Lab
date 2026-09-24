import subprocess
import sys
from pathlib import Path
ROOT = Path.cwd()
for script, lock in [("collect_live.py", "live_collector.lock"), ("ml/live_feature_builder.py", "builder.lock"), ("ml/shadow_runner.py", "shadow.lock")]:
    p = subprocess.Popen([sys.executable, script])
    (ROOT / "ui" / "runtime" / lock).write_text(str(p.pid))
