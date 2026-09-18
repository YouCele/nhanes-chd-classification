"""
Run the whole analysis in order.

    python scripts/run_all.py            all steps
    python scripts/run_all.py 4 5        only scripts 4 and 5

Each step is a separate process, so a failure stops the chain instead of
leaving half-written outputs behind.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("1", "run_01_prepare.py", "phases 1-6: outcome, harmonisation, leakage, cohort, missingness"),
    ("2", "run_02_eda.py", "phase 7: exploratory analysis"),
    ("3", "run_03_statistics.py", "phases 8 and 19: statistics and survey-weighted prevalence"),
    ("4", "run_04_modeling.py", "phases 9-12: split, baselines, tuning, comparison"),
    ("5", "run_05_evaluation.py", "phases 13-15: test evaluation, calibration, interpretation"),
    ("6", "run_06_robustness.py", "phases 16-18: errors, robustness, subgroups"),
    ("7", "run_07_audit.py", "phase 20: audit"),
    ("8", "run_08_report.py", "phase 21: report skeleton with the computed numbers"),
]


def main() -> None:
    wanted = set(sys.argv[1:]) or {s[0] for s in STEPS}
    for num, script, description in STEPS:
        if num not in wanted:
            continue
        print(f"\n########## step {num}: {script} - {description}")
        start = time.time()
        result = subprocess.run([sys.executable, str(HERE / script)])
        if result.returncode != 0:
            print(f"step {num} failed, stopping here")
            raise SystemExit(result.returncode)
        print(f"########## step {num} done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
