from __future__ import annotations

from pathlib import Path
import pandas as pd

from common import ensure_parent

def upsert_parquet(path: Path, df: pd.DataFrame, key_cols: list[str], sort_cols: list[str]) -> None:
    if df.empty:
        return
    ensure_parent(path)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=key_cols, keep="last")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)

def write_parquet(path: Path, df: pd.DataFrame) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
