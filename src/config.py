"""
Central configuration for the NHANES CHD classification project.

Everything that another module might need to know about paths, cycles,
variable names and fixed methodological choices lives here, so that the
analysis code itself stays free of magic strings.

Nothing in this file contains results. Results are always computed from
the raw files at run time.
"""

from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Paths (all relative to the project root, no machine-specific paths)
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The raw directory can be overridden with the NHANES_RAW environment
# variable, which is useful if the XPT files live outside the repository.
RAW_DIR = Path(os.environ.get("NHANES_RAW", PROJECT_ROOT / "data" / "raw"))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

for _d in (PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------

RANDOM_SEED = 2026
N_JOBS = 1          # the project targets a modest 8 GB / single-CPU machine
CV_FOLDS = 5
BOOTSTRAP_N = 1000  # used for confidence intervals on the test set

# ----------------------------------------------------------------------
# Survey cycles
# ----------------------------------------------------------------------

# NHANES suffixes the file name with a letter per cycle.
CYCLES = {
    "G": "2011-2012",
    "H": "2013-2014",
    "I": "2015-2016",
    "J": "2017-2018",
}

# SDDSRVYR is the cycle number stored inside DEMO; used as a sanity check.
EXPECTED_SDDSRVYR = {"G": 7, "H": 8, "I": 9, "J": 10}

FILE_STEMS = ["DEMO", "MCQ", "BPX", "BMX", "TCHOL", "HDL", "TRIGLY"]

# Files that only cover a sub-sample of participants by design.
SUBSAMPLE_FILES = {
    "TRIGLY": "fasting sub-sample (morning session, 8.5+ hour fast)",
}

ID_COL = "SEQN"

# ----------------------------------------------------------------------
# Missing / special response codes
# ----------------------------------------------------------------------

# Standard NHANES codes for questionnaire items with a 1/2 (yes/no) answer.
YESNO_REFUSED = 7
YESNO_DONTKNOW = 9

# Items coded with two digits use 77 / 99 instead.
TWO_DIGIT_REFUSED = 77
TWO_DIGIT_DONTKNOW = 99

# ----------------------------------------------------------------------
# Target
# ----------------------------------------------------------------------

TARGET_SOURCE_VAR = "MCQ160C"
TARGET = "chd"

TARGET_DOC = {
    "variable": "MCQ160C",
    "file": "MCQ (Medical Conditions questionnaire)",
    "question": (
        "Has a doctor or other health professional ever told you that you had "
        "coronary heart disease?"
    ),
    "asked_of": "participants aged 20 years and over",
    "codes": "1 = Yes, 2 = No, 7 = Refused, 9 = Don't know, blank = missing",
    "mapping": "1 -> 1 ; 2 -> 0 ; 7, 9, blank -> excluded (target not usable)",
    "note": (
        "This is self-reported, physician-diagnosed CHD. It is not "
        "angiography-confirmed disease and it is not a clinical gold standard."
    ),
}

# ----------------------------------------------------------------------
# Variable registry
# ----------------------------------------------------------------------
# Each entry: source file stem, a short human name, the meaning, the type,
# and the role it plays in the analysis.
#   role = "id" | "design" | "target" | "predictor" | "leakage_candidate"
#          | "eligibility"
#
# "leakage_candidate" variables are loaded on purpose so that Phase 3 can
# investigate them with real numbers instead of an opinion.

VARIABLES: list[dict] = [
    # --- identifiers and survey design -------------------------------
    dict(var="SEQN", file="DEMO", name="seqn", meaning="participant identifier",
         dtype="id", role="id"),
    dict(var="SDDSRVYR", file="DEMO", name="sddsrvyr", meaning="NHANES cycle number",
         dtype="int", role="design"),
    dict(var="RIDSTATR", file="DEMO", name="exam_status",
         meaning="interview only (1) vs interview + MEC examination (2)",
         dtype="cat", role="eligibility"),
    dict(var="WTINT2YR", file="DEMO", name="wt_int",
         meaning="2-year interview sample weight", dtype="float", role="design"),
    dict(var="WTMEC2YR", file="DEMO", name="wt_mec",
         meaning="2-year MEC examination sample weight", dtype="float", role="design"),
    dict(var="SDMVPSU", file="DEMO", name="psu", meaning="masked variance PSU",
         dtype="int", role="design"),
    dict(var="SDMVSTRA", file="DEMO", name="stratum", meaning="masked variance stratum",
         dtype="int", role="design"),

    # --- demographics -------------------------------------------------
    dict(var="RIDAGEYR", file="DEMO", name="age",
         meaning="age in years at screening (80 = 80 and over)",
         dtype="num", role="predictor"),
    dict(var="RIAGENDR", file="DEMO", name="sex", meaning="sex (1 male, 2 female)",
         dtype="cat", role="predictor"),
    dict(var="RIDRETH3", file="DEMO", name="race_eth",
         meaning="race / Hispanic origin incl. non-Hispanic Asian",
         dtype="cat", role="predictor"),
    dict(var="DMDEDUC2", file="DEMO", name="education",
         meaning="education level, adults 20+", dtype="cat", role="predictor",
         special=[7, 9]),
    dict(var="DMDMARTL", file="DEMO", name="marital",
         meaning="marital status", dtype="cat", role="predictor", special=[77, 99]),
    dict(var="INDFMPIR", file="DEMO", name="income_poverty_ratio",
         meaning="ratio of family income to poverty (top-coded at 5)",
         dtype="num", role="predictor"),
    dict(var="DMDBORN4", file="DEMO", name="born_us",
         meaning="country of birth (1 US, 2 other)", dtype="cat",
         role="predictor", special=[77, 99]),

    # --- target ---------------------------------------------------------
    dict(var="MCQ160C", file="MCQ", name="mcq160c_raw",
         meaning="ever told had coronary heart disease", dtype="cat", role="target"),

    # --- questionnaire predictors (non-cardiovascular) ------------------
    dict(var="MCQ010", file="MCQ", name="asthma_ever",
         meaning="ever told had asthma", dtype="cat", role="predictor"),
    dict(var="MCQ035", file="MCQ", name="asthma_still",
         meaning="still has asthma (asked only of asthma cases)",
         dtype="cat", role="predictor"),
    dict(var="MCQ053", file="MCQ", name="anemia_treated",
         meaning="treated for anemia in the past 3 months", dtype="cat", role="predictor"),
    dict(var="MCQ080", file="MCQ", name="told_overweight",
         meaning="doctor ever said you were overweight", dtype="cat", role="predictor"),
    dict(var="MCQ160A", file="MCQ", name="arthritis",
         meaning="ever told had arthritis", dtype="cat", role="predictor"),
    dict(var="MCQ160G", file="MCQ", name="emphysema",
         meaning="ever told had emphysema", dtype="cat", role="predictor"),
    dict(var="MCQ160K", file="MCQ", name="bronchitis",
         meaning="ever told had chronic bronchitis", dtype="cat", role="predictor"),
    dict(var="MCQ160L", file="MCQ", name="liver_condition",
         meaning="ever told had a liver condition", dtype="cat", role="predictor"),
    dict(var="MCQ160M", file="MCQ", name="thyroid_problem",
         meaning="ever told had a thyroid problem", dtype="cat", role="predictor"),
    dict(var="MCQ160N", file="MCQ", name="gout",
         meaning="ever told had gout", dtype="cat", role="predictor"),
    dict(var="MCQ220", file="MCQ", name="cancer_ever",
         meaning="ever told had cancer or a malignancy", dtype="cat", role="predictor"),
    dict(var="MCQ300A", file="MCQ", name="fam_hist_mi",
         meaning="close relative had a heart attack before age 50",
         dtype="cat", role="predictor"),
    dict(var="MCQ300B", file="MCQ", name="fam_hist_asthma",
         meaning="close relative had asthma", dtype="cat", role="predictor"),
    dict(var="MCQ300C", file="MCQ", name="fam_hist_diabetes",
         meaning="close relative had diabetes", dtype="cat", role="predictor"),

    # --- cardiovascular questionnaire block: leakage candidates ---------
    dict(var="MCQ160B", file="MCQ", name="chf",
         meaning="ever told had congestive heart failure",
         dtype="cat", role="leakage_candidate"),
    dict(var="MCQ160D", file="MCQ", name="angina",
         meaning="ever told had angina / angina pectoris",
         dtype="cat", role="leakage_candidate"),
    dict(var="MCQ160E", file="MCQ", name="heart_attack",
         meaning="ever told had a heart attack (myocardial infarction)",
         dtype="cat", role="leakage_candidate"),
    dict(var="MCQ160F", file="MCQ", name="stroke",
         meaning="ever told had a stroke", dtype="cat", role="leakage_candidate"),
    dict(var="MCQ180C", file="MCQ", name="age_told_chd",
         meaning="age when first told had coronary heart disease",
         dtype="num", role="leakage_candidate", special=[77777, 99999]),

    # --- examination ----------------------------------------------------
    dict(var="BPXSY1", file="BPX", name="sbp1", meaning="systolic BP, reading 1 (mmHg)",
         dtype="num", role="predictor_raw"),
    dict(var="BPXSY2", file="BPX", name="sbp2", meaning="systolic BP, reading 2 (mmHg)",
         dtype="num", role="predictor_raw"),
    dict(var="BPXSY3", file="BPX", name="sbp3", meaning="systolic BP, reading 3 (mmHg)",
         dtype="num", role="predictor_raw"),
    dict(var="BPXSY4", file="BPX", name="sbp4", meaning="systolic BP, reading 4 (mmHg)",
         dtype="num", role="predictor_raw"),
    dict(var="BPXDI1", file="BPX", name="dbp1", meaning="diastolic BP, reading 1 (mmHg)",
         dtype="num", role="predictor_raw"),
    dict(var="BPXDI2", file="BPX", name="dbp2", meaning="diastolic BP, reading 2 (mmHg)",
         dtype="num", role="predictor_raw"),
    dict(var="BPXDI3", file="BPX", name="dbp3", meaning="diastolic BP, reading 3 (mmHg)",
         dtype="num", role="predictor_raw"),
    dict(var="BPXDI4", file="BPX", name="dbp4", meaning="diastolic BP, reading 4 (mmHg)",
         dtype="num", role="predictor_raw"),
    dict(var="BPXPLS", file="BPX", name="pulse",
         meaning="60 second pulse (beats/min)", dtype="num", role="predictor"),
    dict(var="BPXPULS", file="BPX", name="pulse_regular",
         meaning="pulse regular (1) or irregular (2)", dtype="cat", role="predictor"),

    dict(var="BMXBMI", file="BMX", name="bmi", meaning="body mass index (kg/m2)",
         dtype="num", role="predictor"),
    dict(var="BMXWAIST", file="BMX", name="waist", meaning="waist circumference (cm)",
         dtype="num", role="predictor"),
    dict(var="BMXHT", file="BMX", name="height", meaning="standing height (cm)",
         dtype="num", role="predictor"),
    dict(var="BMXWT", file="BMX", name="weight", meaning="weight (kg)",
         dtype="num", role="predictor"),
    dict(var="BMXARMC", file="BMX", name="arm_circ", meaning="arm circumference (cm)",
         dtype="num", role="predictor"),

    # --- laboratory ------------------------------------------------------
    dict(var="LBXTC", file="TCHOL", name="total_chol",
         meaning="total cholesterol (mg/dL)", dtype="num", role="predictor"),
    dict(var="LBDHDD", file="HDL", name="hdl",
         meaning="direct HDL cholesterol (mg/dL)", dtype="num", role="predictor"),
    dict(var="LBXTR", file="TRIGLY", name="triglycerides",
         meaning="triglycerides (mg/dL), fasting sub-sample",
         dtype="num", role="predictor"),
    dict(var="LBDLDL", file="TRIGLY", name="ldl",
         meaning="LDL cholesterol (mg/dL), fasting sub-sample",
         dtype="num", role="predictor"),
]

# Variables that were part of the original research plan but are not present
# in the local file set. The loader checks this list against what it finds so
# the absence is reported instead of silently ignored.
PLANNED_BUT_NOT_DOWNLOADED = {
    "SMQ": "smoking questionnaire (SMQ020, SMQ040) - file not in the local data set",
    "DIQ": "diabetes questionnaire (DIQ010) - file not in the local data set",
    "BPQ": "blood pressure / cholesterol questionnaire (BPQ020, BPQ080) - file not in the local data set",
    "GHB": "glycohemoglobin (LBXGH) - file not in the local data set",
    "GLU": "fasting glucose (LBXGLU) - file not in the local data set",
}

# ----------------------------------------------------------------------
# Plausibility ranges used for data-quality screening.
# Values outside the range are set to missing and counted, never silently kept.
# The ranges are deliberately wide: the aim is to catch coding problems, not
# to remove genuine biological extremes.
# ----------------------------------------------------------------------

PLAUSIBLE_RANGES = {
    "age": (18, 85),
    "sbp": (60, 280),
    "dbp": (20, 160),
    "pulse": (30, 220),
    "bmi": (12, 90),
    "waist": (40, 200),
    "height": (110, 220),
    "weight": (25, 300),
    "arm_circ": (12, 70),
    "total_chol": (50, 700),
    "hdl": (5, 200),
    "triglycerides": (10, 3000),
    "ldl": (10, 500),
    "income_poverty_ratio": (0, 5),
}

# ----------------------------------------------------------------------
# Fixed methodological choices (Phase 9 freezes these)
# ----------------------------------------------------------------------

MIN_AGE = 20          # the CHD question is only asked of adults 20+
REQUIRE_MEC_EXAM = True   # examination data are part of the intended predictors

# Variables whose missingness is a design feature rather than an accident,
# so a missing indicator is meaningful.
FORCE_MISSING_INDICATOR = ["triglycerides", "ldl"]

TEST_SIZE = 0.25

PRIMARY_METRIC = "average_precision"   # PR-AUC, because the classes are imbalanced
SECONDARY_METRIC = "roc_auc"

AGE_BINS = [20, 40, 50, 60, 70, 200]
AGE_LABELS = ["20-39", "40-49", "50-59", "60-69", "70+"]

RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other / multiracial",
}
SEX_LABELS = {1: "Male", 2: "Female"}
EDU_LABELS = {
    1: "Less than 9th grade",
    2: "9-11th grade",
    3: "High school / GED",
    4: "Some college / AA",
    5: "College graduate or above",
}
MARITAL_LABELS = {
    1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated",
    5: "Never married", 6: "Living with partner",
}


def variables_for(file_stem: str) -> list[dict]:
    """Registry entries that come from one raw file."""
    return [v for v in VARIABLES if v["file"] == file_stem]


def raw_columns_for(file_stem: str) -> list[str]:
    """Raw NHANES column names to read from one file (selective loading)."""
    cols = [v["var"] for v in variables_for(file_stem)]
    if ID_COL not in cols:
        cols = [ID_COL] + cols
    return cols


def rename_map(file_stem: str) -> dict:
    return {v["var"]: v["name"] for v in variables_for(file_stem) if v["var"] != ID_COL}


def registry_frame():
    import pandas as pd
    return pd.DataFrame(VARIABLES)
