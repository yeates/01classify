# 01classify

## Usage
1. run `python DataHandler.py` to wash data get **new_train.csv** and **new_test.csv**.
2. then run `python Baseline.py` to make result.csv
3. waiting for 20 minutes.

## Directory Structure
```
project
│   README.md
│   Baseline.py             > Classification
│   DataHandler.py          > Wash data
│   UselessWords.py         > Some global constant
│
└───Data
    │───train.csv
    └───test.csv
```