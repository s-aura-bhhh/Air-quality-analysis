# Air Quality in Indian Cities: Prediction and Profiling

MA515 (FDS) group project. Report draft: [`report/report.md`](report/report.md).

## Data

- Source: Kaggle, "Air Quality Data in India (2015-2020)" (Vopani / Rohan Rao), compiled from the CPCB (Central Pollution Control Board) portal.
- Used: `data/raw/city_day.csv` (committed, 29,531 rows, 26 cities, 2015-01-01 to 2020-07-01).
- Not committed (size): `city_hour.csv`, `station_day.csv`, `station_hour.csv`, `stations.csv`.
- Download date: TODO.

## Layout

```
data/raw/            original file, never edited
notebooks/02-07      one notebook per project step, committed with outputs
src/aq.py            shared pipeline: cleaning decisions, time split, imputers, features, CPCB sub-indices
report/report.md     report draft
report/figures/      all figures (prefix = notebook number)
report/tables/       all result tables as CSV
```

| Notebook | Step |
|---|---|
| `02_data_cleaning` | missingness, duplicates, outliers, AQI/bucket consistency, drop rules |
| `03_eda` | distributions, class balance, seasonality, lockdown, correlations, autocorrelation |
| `04_preprocessing` | time split, log/standardise, imputation comparison (median / KNN / SVD), features |
| `05_rq1_regression` | MLR, PCR, KNN, RF, ANN; binding pollutant; tomorrow's AQI; no-Ahmedabad check |
| `06_rq2_classification` | baselines, LogReg, LDA, tree, RF, KNN, ANN; ROC/PR; thresholds; error analysis |
| `07_rq3_profiles` | city features, PCA = SVD check, k-means + hierarchical, profiles, stability |

## Key decisions

- Train 2015-2018, validation 2019, test 2020 (Jan-Jun). Everything fitted on train only.
- Modelling cities: 17 with >= 200 AQI days in 2015-2018. Mumbai and Xylene dropped (> 50% missing).
- Ahmedabad kept; its CO readings look like a sensor problem, so every analysis is repeated without it.
- Imputer: city-centred KNN (k = 80), chosen by a masked-value experiment.

## Run

```
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order 02 to 07. Steps 04-07 import `src/aq.py`. In Google Colab, the first cell of notebooks 04-07 clones this repository; this works only if the repository is public.
