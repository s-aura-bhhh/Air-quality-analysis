"""Shared data pipeline for the air-quality project.

Every notebook from Step 4 on imports from here, so the cleaning decisions,
the time split, the imputation and the features are defined in one place.

Order of operations (prevents leakage):
    load + Step 2 decisions -> time split -> transform/standardise (train stats)
    -> impute (fitted on train) -> features -> targets
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "city_day.csv"

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"]
BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
UNHEALTHY_AQI = 200          # AQI > 200 means Poor or worse
TRAIN_END = pd.Timestamp("2019-01-01")
VAL_END = pd.Timestamp("2020-01-01")
LOCKDOWN = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-31"))
SEED = 42


# ---------------------------------------------------------------- loading
def aqi_to_bucket(aqi):
    return pd.cut(aqi, [-np.inf, 50, 100, 200, 300, 400, np.inf], labels=BUCKET_ORDER).astype("object")


def load_checked(path=RAW):
    """city_day.csv with the Step 2 decisions applied."""
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values(["City", "Date"]).reset_index(drop=True)
    df[POLLUTANTS] = df[POLLUTANTS].mask(df[POLLUTANTS] < 0)
    df = df.drop(columns=["Xylene"])
    df = df[df["City"] != "Mumbai"].copy()
    df["AQI_Bucket"] = aqi_to_bucket(df["AQI"])
    return df.reset_index(drop=True)


def model_cities(df, min_train_aqi_days=200):
    n = df[df["Date"] < TRAIN_END].groupby("City")["AQI"].count()
    return sorted(n[n >= min_train_aqi_days].index)


def period(dates):
    return pd.Series(np.select([dates < TRAIN_END, dates < VAL_END], ["train", "val"], "test"),
                     index=dates.index)


def in_lockdown(dates):
    return (dates >= LOCKDOWN[0]) & (dates <= LOCKDOWN[1])


# ---------------------------------------------------------------- transforms
class LogStandardiser:
    """log1p where it reduces |skew| on train, then standardise with train mean/std."""

    def fit(self, X):
        self.cols = list(X.columns)
        self.log_cols = [c for c in self.cols
                         if abs(np.log1p(X[c]).skew()) < abs(X[c].skew())]
        T = self._log(X)
        self.mean_, self.std_ = T.mean(), T.std()
        return self

    def _log(self, X):
        T = X.copy()
        T[self.log_cols] = np.log1p(T[self.log_cols])
        return T

    def transform(self, X):
        return (self._log(X[self.cols]) - self.mean_) / self.std_

    def inverse_transform(self, Z):
        T = Z * self.std_ + self.mean_
        T[self.log_cols] = np.expm1(T[self.log_cols])
        return T


# ---------------------------------------------------------------- imputers (on standardised data)
class CityMedianImputer:
    """Baseline: each city's train median; global train median if the city never measured it."""

    def fit(self, Z, city):
        self.global_ = Z.median()
        self.city_ = Z.groupby(city.values).median()
        return self

    def transform(self, Z, city):
        med = self.city_.reindex(city.values).set_axis(Z.index)
        return Z.fillna(med).fillna(self.global_)


class KNNImputerDF:
    def __init__(self, k=10):
        self.k = k

    def fit(self, Z, city=None):
        self.imp = KNNImputer(n_neighbors=self.k).fit(Z.values)
        return self

    def transform(self, Z, city=None):
        out = self.imp.transform(Z.values)
        return pd.DataFrame(out, index=Z.index, columns=Z.columns)


class SVDImputer:
    """Low-rank (rank-k) iterative SVD imputation.

    fit: start with missing = 0 (the column mean after standardising), then repeat
         Z_hat = U_k S_k V_k^T and overwrite only the missing cells, until it converges.
    transform (new rows): for each row, least-squares fit of its observed values on
         the k right singular vectors, then predict its missing values from that fit.
    """

    def __init__(self, k=3, max_iter=200, tol=1e-5, ridge=1e-3):
        self.k, self.max_iter, self.tol, self.ridge = k, max_iter, tol, ridge

    def fit(self, Z, city=None):
        X = Z.values.copy()
        miss = np.isnan(X)
        X[miss] = 0.0
        for it in range(self.max_iter):
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            R = (U[:, :self.k] * s[:self.k]) @ Vt[:self.k]
            delta = np.sum((X[miss] - R[miss]) ** 2) / max(np.sum(X[miss] ** 2), 1e-12)
            X[miss] = R[miss]
            if delta < self.tol:
                break
        self.n_iter_ = it + 1
        self.V_ = Vt[:self.k].T                     # p x k
        self.singular_values_ = s
        return self

    def transform(self, Z, city=None):
        X = Z.values.copy()
        miss = np.isnan(X)
        V, lam = self.V_, self.ridge
        patterns = pd.Series([m.tobytes() for m in miss])
        for _, idx in patterns.groupby(patterns).groups.items():
            m = miss[idx[0]]
            if not m.any():
                continue
            o = ~m
            if not o.any():
                X[idx] = 0.0
                continue
            Vo = V[o]
            coef = np.linalg.solve(Vo.T @ Vo + lam * np.eye(V.shape[1]), Vo.T @ X[np.ix_(idx, o)].T)
            X[np.ix_(idx, m)] = (V[m] @ coef).T
        return pd.DataFrame(X, index=Z.index, columns=Z.columns)


class CityCentred:
    """Subtract each city's train median, impute the residuals with `base`, add the median back.
    Combines the city level (what the median imputer knows) with the correlation between
    pollutants on the same day (what KNN/SVD know)."""

    def __init__(self, base):
        self.base = base

    def fit(self, Z, city):
        self.med = CityMedianImputer().fit(Z, city)
        self.base.fit(Z - self._centre(Z, city), city)
        return self

    def _centre(self, Z, city):
        return self.med.transform(pd.DataFrame(np.nan, index=Z.index, columns=Z.columns), city)

    def transform(self, Z, city):
        c = self._centre(Z, city)
        return self.base.transform(Z - c, city) + c


IMPUTERS = {
    "city_median": lambda: CityMedianImputer(),
    "knn": lambda: KNNImputerDF(k=10),
    "svd": lambda: SVDImputer(k=1),
    "city_knn": lambda: CityCentred(KNNImputerDF(k=80)),
    "city_svd": lambda: CityCentred(SVDImputer(k=1)),
}
CHOSEN_IMPUTER = "city_knn"     # lowest masked RMSE in notebook 04 (two random maskings)


def masking_experiment(Z, city, is_train, imputers, frac=0.1, seed=SEED):
    """Hide `frac` of the observed non-train cells, impute with imputers fitted on train,
    return RMSE (standardised units) overall and per pollutant."""
    rng = np.random.default_rng(seed)
    Zv = Z[~is_train]
    obs = ~Zv.isna().values
    hide = obs & (rng.random(obs.shape) < frac)
    Zh = Zv.mask(hide)
    rows = {}
    for name, make in imputers.items():
        imp = make().fit(Z[is_train], city[is_train])
        filled = imp.transform(Zh, city[~is_train]).values
        err = (filled - Zv.values)[hide]
        per_col = [np.sqrt(np.mean((filled[:, j] - Zv.values[:, j])[hide[:, j]] ** 2))
                   for j in range(Z.shape[1])]
        rows[name] = [np.sqrt(np.mean(err ** 2))] + per_col
    return pd.DataFrame(rows, index=["overall"] + list(Z.columns)).T


# ---------------------------------------------------------------- dataset builder
def build_dataset(df=None, cities=None, imputer=CHOSEN_IMPUTER, exclude_cities=()):
    """Returns (data, info). `data` has one row per city-day of the chosen cities with:
    Z_<pollutant> (imputed, log-standardised), feature columns, targets and `period`.
    Everything that is fitted (transform, imputer, AQI fill values) uses train rows only."""
    if df is None:
        df = load_checked()
    if cities is None:
        cities = model_cities(df)
    cities = [c for c in cities if c not in exclude_cities]
    d = df[df["City"].isin(cities)].sort_values(["City", "Date"]).reset_index(drop=True)
    d["period"] = period(d["Date"])
    tr = d["period"] == "train"

    ls = LogStandardiser().fit(d.loc[tr, POLLUTANTS])
    Z = ls.transform(d[POLLUTANTS])
    imp = IMPUTERS[imputer]().fit(Z[tr], d.loc[tr, "City"])
    Zi = imp.transform(Z, d["City"])
    zcols = [f"Z_{p}" for p in POLLUTANTS]
    d[zcols] = Zi.values
    d[[f"miss_{p}" for p in POLLUTANTS]] = Z.isna().values.astype(int)

    # AQI as a feature (RQ2): log, standardised with train stats; missing -> city train median
    la = np.log1p(d["AQI"])
    mu, sd = la[tr].mean(), la[tr].std()
    d["aqi_z"] = (la - mu) / sd
    city_med = d.loc[tr].groupby("City")["aqi_z"].median()
    d["aqi_missing"] = d["aqi_z"].isna().astype(int)
    d["aqi_z_filled"] = d["aqi_z"].fillna(d["City"].map(city_med)).fillna(0.0)

    g = d.groupby("City")
    for c in ["Z_PM2.5", "Z_PM10", "Z_NO2", "aqi_z_filled"]:
        d[f"lag1_{c}"] = g[c].shift(1)
    d["roll7_Z_PM2.5"] = g["Z_PM2.5"].transform(lambda s: s.rolling(7, min_periods=1).mean())

    m = d["Date"].dt.month
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    dw = d["Date"].dt.dayofweek
    d["dow_sin"], d["dow_cos"] = np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)
    city_dummies = pd.get_dummies(d["City"], prefix="city", dtype=int)
    d = pd.concat([d, city_dummies], axis=1)

    # targets
    d["y_log_aqi"] = la                                         # RQ1, today
    d["AQI_next"] = g["AQI"].shift(-1)
    d["y_log_aqi_next"] = np.log1p(d["AQI_next"])
    d["unhealthy_today"] = (d["AQI"] > UNHEALTHY_AQI).astype(float).where(d["AQI"].notna())
    d["y_unhealthy_next"] = (d["AQI_next"] > UNHEALTHY_AQI).astype(float).where(d["AQI_next"].notna())
    d["lockdown"] = in_lockdown(d["Date"])

    rq1_features = zcols
    rq2_features = (zcols + ["aqi_z_filled", "aqi_missing", "lag1_Z_PM2.5", "lag1_Z_PM10",
                             "lag1_Z_NO2", "lag1_aqi_z_filled", "roll7_Z_PM2.5",
                             "month_sin", "month_cos", "dow_sin", "dow_cos"]
                    + list(city_dummies.columns))
    first_day = g.cumcount() == 0
    d = d[~first_day].reset_index(drop=True)          # no lag available
    info = dict(cities=cities, log_standardiser=ls, imputer=imp,
                rq1_features=rq1_features, rq2_features=rq2_features,
                aqi_log_mean=mu, aqi_log_std=sd)
    return d, info


# ---------------------------------------------------------------- CPCB sub-indices
# Breakpoints of the Indian National AQI (CPCB 2014). Concentrations in ug/m3, CO in mg/m3.
# Above the last breakpoint the scale is extended linearly (as in the Kaggle dataset).
_BP = {
    "PM10":  [0, 50, 100, 250, 350, 430, 510],
    "PM2.5": [0, 30, 60, 90, 120, 250, 380],
    "NO2":   [0, 40, 80, 180, 280, 400, 520],
    "O3":    [0, 50, 100, 168, 208, 748, 1288],
    "CO":    [0, 1, 2, 10, 17, 34, 51],
    "SO2":   [0, 40, 80, 380, 800, 1600, 2400],
    "NH3":   [0, 200, 400, 800, 1200, 1800, 2400],
}
_IDX = [0, 50, 100, 200, 300, 400, 500]


def sub_index(x, pollutant):
    return pd.Series(np.interp(x, _BP[pollutant], _IDX, right=np.nan), index=x.index).where(
        x <= _BP[pollutant][-1],
        400 + (x - _BP[pollutant][-2]) * 100 / (_BP[pollutant][-1] - _BP[pollutant][-2]))


def sub_indices(df):
    """Daily-mean approximation of CPCB sub-indices (official AQI uses 8-h max for CO and O3)."""
    return pd.DataFrame({p: sub_index(df[p], p) for p in _BP}, index=df.index)
