"""
Finding and reading the raw NHANES XPT files.

Two ideas drive this module:

1. Nothing is assumed about the folder layout. The files are located by
   searching for <STEM>_<LETTER>.XPT under the raw directory, so the code
   works whether the files sit in one folder or in one folder per cycle.
2. Only the columns listed in the registry are kept. The XPT format has to
   be read as a whole, but dropping the unused columns straight away keeps
   the memory footprint small.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .utils import assert_unique_id, fail, note, step


def find_file(stem: str, cycle_letter: str) -> Path | None:
    """Locate <stem>_<letter>.XPT anywhere under RAW_DIR (case-insensitive)."""
    target = f"{stem}_{cycle_letter}.xpt".lower()
    for path in config.RAW_DIR.rglob("*"):
        if path.is_file() and path.name.lower() == target:
            return path
    return None


def load_raw(stem: str, cycle_letter: str, columns: list[str] | None = None) -> pd.DataFrame | None:
    """Read one raw file and keep only the requested columns."""
    path = find_file(stem, cycle_letter)
    if path is None:
        return None
    df = pd.read_sas(path, format="xport")
    df.columns = [c.upper() for c in df.columns]
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    if config.ID_COL in df.columns:
        df[config.ID_COL] = df[config.ID_COL].astype("int64")
    return df


def file_inventory() -> pd.DataFrame:
    """
    Check which of the required files exist, how many rows they have, whether
    SEQN is unique inside each file, and which registry variables are present.

    This repeats a small part of Phase 0 on purpose: it is cheap, and it means
    every later run starts from a verified file set instead of a memory of one.
    """
    rows = []
    for letter, cycle in config.CYCLES.items():
        for stem in config.FILE_STEMS:
            path = find_file(stem, letter)
            if path is None:
                rows.append(dict(cycle=cycle, file=stem, found=False, path="",
                                 n_rows=0, n_cols=0, unique_seqn=False,
                                 registry_vars_present="", registry_vars_missing=""))
                continue
            df = load_raw(stem, letter)
            wanted = [v["var"] for v in config.variables_for(stem)]
            present = [v for v in wanted if v in df.columns]
            missing = [v for v in wanted if v not in df.columns]
            rows.append(dict(
                cycle=cycle, file=stem, found=True,
                path=str(path.relative_to(config.RAW_DIR)),
                n_rows=len(df), n_cols=df.shape[1],
                unique_seqn=bool(df[config.ID_COL].is_unique),
                registry_vars_present=";".join(present),
                registry_vars_missing=";".join(missing),
            ))
    return pd.DataFrame(rows)


def check_inventory(inv: pd.DataFrame) -> None:
    """Fail loudly if a required file is missing or breaks the one-row rule."""
    step("checking the raw file inventory")
    missing_files = inv.loc[~inv["found"], ["cycle", "file"]]
    if len(missing_files):
        fail("missing raw files:\n" + missing_files.to_string(index=False))

    bad_id = inv.loc[~inv["unique_seqn"], ["cycle", "file"]]
    if len(bad_id):
        fail("SEQN is not unique in:\n" + bad_id.to_string(index=False))

    note(f"{len(inv)} files found, SEQN unique in all of them")

    # Report, but do not fail on, registry variables that a cycle does not have.
    gaps = inv.loc[inv["registry_vars_missing"] != "", ["cycle", "file", "registry_vars_missing"]]
    if len(gaps):
        note("variables listed in the registry but absent in some cycles:")
        for _, r in gaps.iterrows():
            note(f"   {r['cycle']} {r['file']}: {r['registry_vars_missing']}")

    for stem, reason in config.PLANNED_BUT_NOT_DOWNLOADED.items():
        note(f"not available locally: {stem} -> {reason}")


def one_row_per_participant(df: pd.DataFrame, label: str) -> None:
    assert_unique_id(df, where=label)
