# Raw data

The XPT files are not stored in this repository. NHANES data are public but it
is cleaner to download them from the source than to redistribute them.

Download from the CDC, one folder per cycle (any layout works, the loader
searches recursively):

- 2011-2012 (suffix `_G`): DEMO_G, MCQ_G, BPX_G, BMX_G, TCHOL_G, HDL_G, TRIGLY_G
- 2013-2014 (suffix `_H`): the same seven files with `_H`
- 2015-2016 (suffix `_I`): the same seven files with `_I`
- 2017-2018 (suffix `_J`): the same seven files with `_J`

Source: https://wwwn.cdc.gov/nchs/nhanes/ (Questionnaire, Examination and
Laboratory data pages for each cycle).

Put them under `data/raw/`, or point the code somewhere else:

    export NHANES_RAW=/path/to/xpt/files
    python scripts/run_all.py

`scripts/run_01_prepare.py` checks that all 28 files exist and that SEQN is
unique in each one before doing anything else.
