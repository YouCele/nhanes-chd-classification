"""Small helpers shared by every stage: logging, saving, simple checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

_WIDTH = 78


def banner(text: str) -> None:
    print("\n" + "=" * _WIDTH)
    print(text)
    print("=" * _WIDTH)


def step(text: str) -> None:
    print(f"\n-- {text}")


def note(text: str) -> None:
    print(f"   {text}")


def fail(text: str) -> None:
    """Stop the run. Used when continuing would mean analysing broken data."""
    print(f"\n!! BLOCKING PROBLEM: {text}", file=sys.stderr)
    raise SystemExit(1)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = config.OUTPUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    note(f"saved {path.relative_to(config.PROJECT_ROOT)}  ({len(df)} rows)")
    return path


def save_json(obj, name: str) -> Path:
    path = config.OUTPUT_DIR / name
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    note(f"saved {path.relative_to(config.PROJECT_ROOT)}")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_fig(fig, name: str) -> Path:
    path = config.FIGURE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    note(f"saved {path.relative_to(config.PROJECT_ROOT)}")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def read_processed(name: str) -> pd.DataFrame:
    """Read an intermediate file, falling back to CSV when parquet is absent."""
    path = config.PROCESSED_DIR / name
    if not path.exists() and path.suffix == ".parquet":
        path = path.with_suffix(".csv")       # pyarrow may not be installed
    if not path.exists():
        fail(f"{path} not found. Run the earlier script first.")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def write_processed(df: pd.DataFrame, name: str) -> Path:
    path = config.PROCESSED_DIR / name
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception:                      # pyarrow missing -> fall back
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
    note(f"saved {path.relative_to(config.PROJECT_ROOT)}  shape={df.shape}")
    return path


def assert_unique_id(df: pd.DataFrame, col: str = config.ID_COL, where: str = "") -> None:
    if col not in df.columns:
        fail(f"{col} missing in {where}")
    if df[col].isna().any():
        fail(f"{col} contains missing values in {where}")
    dup = int(df[col].duplicated().sum())
    if dup:
        fail(f"{dup} duplicated {col} values in {where}; one row per participant is assumed")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Float64 -> float32 for numeric columns, to keep memory low."""
    out = df.copy()
    for c in out.columns:
        if c == config.ID_COL:
            continue
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype("float32")
    return out
