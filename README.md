# Air Quality in Indian Cities: Prediction and Profiling

MA515 (FDS) group project.

## Data

- Source: Kaggle, "Air Quality Data in India (2015-2020)" (Vopani), originally from the CPCB (Central Pollution Control Board) portal.
- Main file: `data/raw/city_day.csv` (committed).
- Other files (`city_hour.csv`, `station_day.csv`, `station_hour.csv`, `stations.csv`) stay local and out of git because of their size.
- Download date: TODO (fill in).

## Layout

```
data/raw/        original files, never edited
data/interim/    outputs of cleaning steps (regenerated, not committed)
notebooks/       one notebook per project step (02_data_cleaning.ipynb, ...)
src/             shared functions reused across notebooks
report/          figures/, tables/, final report
```

## Run

```
pip install -r requirements.txt
jupyter notebook notebooks/02_data_cleaning.ipynb
```
