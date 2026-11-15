# Air Quality in Indian Cities: Prediction and Profiling

MA515 (FDS) Group Project. Code: `notebooks/02`–`07`, shared pipeline `src/aq.py`. Figures referenced below are in `report/figures/` and tables in `report/tables/`.

> **Draft status.** All numbers come from the executed notebooks. Sections marked **[TEAM]** need your input. Rewrite the interpretation in your own words before you submit.

## 1. Problem formulation

Air pollution is one of India's largest public-health risks, and monitoring stations publish daily pollutant readings for many cities. This project asks three questions of those readings: what drives the Air Quality Index (AQI), whether an unhealthy day can be predicted one day ahead, and whether cities fall into distinct pollution profiles.

- **RQ1.** Which pollutants drive AQI, and how well can AQI be predicted from them? (regression, PCR)
- **RQ2.** Can we predict whether tomorrow will be unhealthy (AQI > 200, i.e. "Poor" or worse) from today's readings? (classification, ROC-AUC)
- **RQ3.** Do cities fall into distinct pollution profiles, and how many underlying factors explain the variation? (PCA, SVD, clustering)

## 2. Data

**Source.** Kaggle, "Air Quality Data in India (2015–2020)", compiled from the Central Pollution Control Board (CPCB) portal. File used: `city_day.csv`. Download date: **[TEAM]**.

**Size.** 29,531 rows (city-days), 26 cities, 1 Jan 2015 – 1 Jul 2020. Columns: City, Date, 12 pollutants (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene; all in µg/m³ except CO in mg/m³), AQI and AQI_Bucket.

**Quality checks (notebook 02).**
- There are no duplicate city-dates and no negative values. Within each city the dates are continuous, with no gaps, so a one-row shift is a one-day shift.
- Cities start reporting at very different times. Seven cities begin in 2015. Ernakulam, Kochi and Aizawl begin only in 2020.
- Missing values (% of rows): Xylene 61.3, PM10 37.7, NH3 35.0, Toluene 27.2, Benzene 19.0, AQI 15.9, PM2.5 15.6, and 7–14 for the rest. Some cities never measured certain pollutants; for example, Lucknow has no PM10 at all (`missing_heatmap_city.png`).
- AQI and AQI_Bucket are missing on exactly the same 4,681 rows and never contradict each other. The bucket can be recovered from AQI, but AQI cannot be recovered from the bucket, so there was nothing to repair.
- Outliers were flagged with a per-city IQR rule on the log scale, not deleted. In 13 of 26 cities, at least 80% of the top-1% PM2.5 days fall in October–January (Delhi 90%, Lucknow 80%, Patna 100%). This shows that extreme values are seasonal pollution episodes, not sensor errors. Gurugram is the exception, at 19%.
- There are many exact zeros (CO 2,328; Benzene 3,802). This is likely sensor dropout. These values were kept and are noted as a limitation.

**Decisions.**
1. Drop Xylene (more than 50% missing).
2. Drop Mumbai (61% of its AQI values missing; AQI is the target).
3. For RQ1 and RQ2, use the **17 cities with at least 200 AQI days in the training years 2015–2018**. A city that is absent from training cannot be learned by a model that uses city as a feature.
4. **Ahmedabad.** Its median CO is 16.2 mg/m³, against 1.85 for the next-highest city. Its CO rises from 8 (2015) to 29 (2018) and falls to 3.4 (2020), while its PM2.5 is ordinary. Because AQI is a maximum over sub-indices, its AQI (median 384, maximum 2,049) is mostly a CO reading. We consider a calibration problem likely. The team decided to keep Ahmedabad, and every analysis is repeated without it (sections 6.4, 7.4, 8.3).

## 3. Exploratory data analysis (notebook 03)

| # | Finding | Evidence | Decision |
|---|---|---|---|
| 1 | Strong right skew | PM2.5 skew 3.34 → 0.06 after log1p; Benzene 20.5 → 1.2; CO 8.6 → 2.6 | log1p transform |
| 2 | Pollutants correlated in groups | Spearman PM2.5–PM10 0.88, Benzene–Toluene 0.77, NO–NOx 0.74; AQI–PM10 0.87, AQI–PM2.5 0.84 | PCR (RQ1), PCA (RQ3), VIF check |
| 3 | Imbalanced, drifting target | 26.7% of days Poor or worse; positive rate 33.5% (2015–18) → 24.6% (2019) → 12.4% (2020) | ROC-AUC, PR-AUC, F2; threshold set on validation |
| 4 | Strong seasonality | Monthly mean AQI 252 (Nov), 241 (Jan) vs 115 (Jul) | cyclic month features |
| 5 | Day-of-week effect is negligible | about ±4 AQI points, versus a seasonal swing of about 140 | kept as a cheap feature; not interpreted |
| 6 | Lockdown natural experiment | Mar–May 2020 vs 2019, median over cities: AQI −29%, PM2.5 −36%, NO2 −39%, CO −24%, O3 +12% | report test results also without Mar–May 2020 |
| 7 | Cities differ strongly | median AQI: Ahmedabad, Delhi, Patna, Gurugram, Lucknow above or near 200; southern cities about 60–100 | city one-hot (RQ2), profiles (RQ3) |
| 8 | AQI is highly persistent | lag-1 correlation high; P(unhealthy tomorrow \| unhealthy today) = 0.88, versus 0.06 if today is healthy | lag features; persistence baselines |
| 9 | City mix changes over time | 2015 has about 7 cities, dominated by Delhi and Ahmedabad | city-adjusted yearly trend; fixed city panel |

The rise in O3 during the lockdown while NO2 fell is consistent with less NO being available to remove ozone. NO2 falling the most is consistent with reduced traffic. Weather differs between 2019 and 2020, so this comparison is suggestive, not causal.

## 4. Methodology

### 4.1 Pipeline and leakage control
All modelling uses the 17 cities that have at least 200 days with an AQI value in 2015–2018. Cities that only start reporting in 2019–2020 cannot be learned by a model that uses city as a feature, so they are kept for EDA and RQ3 only. The data is split by time: **train 2015–2018, validation 2019, test 1 Jan–1 Jul 2020**. Everything that is estimated from data (log decision, means and standard deviations, imputer, fill values) is fitted on the training rows only, and then applied unchanged to validation and test. Hyperparameters and decision thresholds are chosen on validation. Each final model is evaluated once on test. The test period includes the COVID-19 lockdown, so test results are also reported with March–May 2020 removed.

### 4.2 Transformation and imputation
A pollutant is log-transformed, x ↦ log(1 + x), if that reduces its absolute skewness on the training data, which was true for all 11 pollutants. It is then standardised, z = (x − μ_train) / σ_train.

Missing pollutant values are imputed in this standardised space. We compared five families of imputer:
- **per-city median**: each city's training median;
- **KNN**: the mean of the k nearest training rows, using the observed coordinates to measure distance;
- **low-rank SVD**: start the missing cells at 0 (the column mean). Then repeat: compute Z = UΣVᵀ, form the rank-k reconstruction Ẑ = U_k Σ_k V_kᵀ, and overwrite only the missing cells with the values of Ẑ, until the change is small. A new row is imputed by solving min_c ‖z_o − V_{k,o} c‖² + λ‖c‖² over its observed entries o, then setting z_m = V_{k,m} c;
- **city-centred KNN / SVD**: subtract each city's median, impute the remaining residual with KNN or SVD, then add the median back.

The methods were compared by hiding 10% of the *observed* validation values, imputing them with each method (fitted on train), and computing the RMSE between imputed and true values.

### 4.3 Multiple linear regression
With design matrix X (n × p, including an intercept column) and target y = log(1 + AQI), ordinary least squares minimises ‖y − Xβ‖². Setting the gradient to zero gives the normal equations XᵀXβ = Xᵀy, so

  β̂ = (XᵀX)⁻¹Xᵀy,  Var(β̂) = σ²(XᵀX)⁻¹.

Collinearity makes XᵀX close to singular, which inflates Var(β̂). The variance inflation factor of predictor j is VIF_j = 1/(1 − R_j²), where R_j² is the R² from regressing predictor j on all the other predictors.

### 4.4 PCA, SVD and principal component regression
Let X be the centred n × p data matrix, with thin SVD X = UΣVᵀ. The sample covariance is

  S = XᵀX/(n − 1) = V (Σ²/(n − 1)) Vᵀ.

So the columns of V (the right singular vectors) are the eigenvectors of S, i.e. the principal directions or loadings, and the eigenvalues are λ_j = σ_j²/(n − 1). The principal component scores are T = XV = UΣ. Component j explains a share λ_j / Σ_i λ_i of the total variance. PCR regresses y on the first k columns of T, which are uncorrelated by construction. In notebook 07, PCA computed with scikit-learn and with numpy's SVD are checked to give the same eigenvalues, loadings and scores (up to sign).

### 4.5 Logistic regression and linear classifiers
For a binary label y_i ∈ {0, 1}, logistic regression models p_i = P(y_i = 1 | x_i) = 1/(1 + e^{−(β₀ + xᵢᵀβ)}). It minimises the regularised, class-weighted log-loss

  L(β) = −Σ_i w_{y_i} [ y_i log p_i + (1 − y_i) log(1 − p_i) ] + (1/(2C)) ‖β‖²,

where w_c = n / (2 n_c) gives the rarer class more weight ("balanced") and C sets the strength of the L2 penalty. exp(β_j) is the odds ratio for a one-standard-deviation increase in feature j. LDA assumes Gaussian classes with a shared covariance Σ, which gives a linear decision rule δ_c(x) = xᵀΣ⁻¹μ_c − ½μ_cᵀΣ⁻¹μ_c + log π_c. Σ is shrunk towards a diagonal matrix (Ledoit–Wolf) to stabilise the estimate.

### 4.6 Trees, forests, KNN, neural networks
A decision tree splits on the feature and threshold that most reduce Gini impurity, 1 − Σ_c p_c². A random forest averages many trees, each grown on a bootstrap sample and considering a random subset of features at each split. Averaging reduces variance. KNN predicts from the k nearest training points in the standardised feature space. The ANN is a multilayer perceptron with ReLU hidden layers, trained with Adam and an L2 penalty α, with early stopping on an internal validation fraction.

### 4.7 Evaluation
Regression is evaluated with RMSE, MAE and R² on the AQI scale. RQ2 uses the ROC-AUC, which is the probability that a randomly chosen unhealthy day receives a higher score than a randomly chosen healthy day, and therefore does not depend on the positive rate. It also uses the precision-recall AUC (average precision), together with precision, recall, F1 and F2 at a threshold chosen on validation. Here F_β = (1 + β²)·P·R / (β²P + R), and β = 2 weights recall more heavily because a missed unhealthy day costs more than a false alarm.

### 4.8 Clustering
k-means minimises the within-cluster sum of squares

  J = Σ_{k=1}^{K} Σ_{i ∈ C_k} ‖x_i − μ_k‖²,

by alternating two steps: assign each point to its nearest centroid, then recompute each centroid μ_k as the mean of its points. This is run from 50 random starts. Ward hierarchical clustering repeatedly merges the pair of clusters whose merge increases J the least. K is chosen with the silhouette score s_i = (b_i − a_i)/max(a_i, b_i), where a_i is the mean distance from point i to the other points in its own cluster and b_i is its mean distance to the nearest other cluster. Two partitions are compared with the adjusted Rand index.

## 5. Preprocessing results (notebook 04)

- **Split.** Train 15,978 rows, validation 6,205, test 3,111 (17 cities in each).
- **Transform.** log1p reduced skewness on train for all 11 pollutants.
- **Imputation experiment** (RMSE on hidden validation values, standardised units; two random maskings):

| Method | RMSE (seed 1) | RMSE (seed 2) |
|---|---|---|
| city-centred KNN, k = 80 | **0.729** | **0.748** |
| city-centred KNN, k = 40 | 0.751 | 0.768 |
| city-centred SVD, rank 1 | 0.754 | 0.780 |
| KNN, k = 40 | 0.792 | 0.807 |
| SVD, rank 1 | 0.811 | 0.824 |
| per-city median (baseline) | 0.818 | 0.835 |
| SVD, rank 3 | 0.887 | 0.911 |

Plain KNN and SVD barely beat the city median, because the largest source of variation is *which city* a row comes from. Removing city medians first and then exploiting same-day correlations between pollutants gives the best result. SVD ranks above 2 overfit. Even so, the gain is modest (0.82 → 0.73): a missing pollutant value is only partly predictable from the other pollutants on the same day. We chose city-centred KNN with k = 80.

## 6. RQ1: what drives AQI (notebook 05)

### 6.1 Model comparison (target log AQI; metrics on the AQI scale)

| Model | Val RMSE | Test RMSE | Test MAE | Test R² | Test R² (no lockdown) |
|---|---|---|---|---|---|
| MLR | 59.2 | 35.7 | 22.4 | 0.844 | 0.860 |
| PCR (k = 10) | 58.8 | 35.3 | 22.3 | 0.847 | 0.863 |
| KNN (k = 10) | 54.3 | 36.2 | 22.0 | 0.839 | 0.851 |
| Random forest (100 trees, depth 12) | **49.5** | **29.5** | **17.1** | **0.894** | **0.909** |
| ANN (64 hidden units) | 51.4 | 33.0 | 19.9 | 0.866 | 0.886 |

The random forest is best, cutting test RMSE by 17% compared with MLR. AQI is the *maximum* of piecewise-linear sub-indices, so it is a non-linear function of the pollutants, and tree models suit that shape. Test errors are lower than validation errors because 2020 AQI levels are lower.

### 6.2 Collinearity and PCR
VIFs are all below 2.7 (largest: Toluene 2.62, NO 2.36, PM10 2.32). That is lower than the plan expected. In the modelling data many PM10 values are imputed (Lucknow never measured PM10), which weakens the PM2.5–PM10 link, and the log transform reduces the influence of shared extremes; this is a likely explanation, not a tested one. Cross-validation chose k = 10 of 11 components, and PCR matches MLR (RMSE 35.3 vs 35.7). With n ≫ p and only moderate collinearity, removing collinearity does not improve prediction. Collinearity mainly affects how stable individual coefficients are, not predictive accuracy.

### 6.3 Drivers
- MLR standardised coefficients: PM2.5 0.36, CO 0.27, PM10 0.16, all others below 0.05.
- Random forest permutation importance (validation): PM2.5 0.66, CO 0.36, PM10 0.13, O3 0.04.

Both methods agree on the ranking: **PM2.5 first, then CO and PM10.** Gaseous pollutants contribute little.

**The binding pollutant.** We recomputed each pollutant's CPCB sub-index from the daily values and took the pollutant with the highest sub-index. This approximation matches the dataset's AQI at r = 0.88, with the same bucket on 70% of days; the official AQI uses hourly rolling windows. By season, PM2.5 binds on 63% of winter days and 57% of post-monsoon days, while PM10 binds most often in summer (48%) and the monsoon (42%). On *unhealthy* days, PM2.5 is the binding pollutant in almost every city (Lucknow 99%, Patna 98%, Kolkata 99%, Gurugram 94%, Delhi 71% with PM10 26%). The exceptions are Ahmedabad (CO 89%), Bengaluru (CO 73% of its 41 unhealthy days), and Talcher and Jorapokhar (PM10 about 53%).

### 6.4 Without Ahmedabad
CO's MLR coefficient halves (0.269 → 0.135) and PM2.5's rises (0.363 → 0.395). So "CO drives AQI" in the full model is largely one city's suspect sensor. Test R² is 0.827 (MLR) and 0.886 (random forest).

### 6.5 Non-circular version: tomorrow's AQI
Using today's data, lags and city as features:

| | Test RMSE | Test R² |
|---|---|---|
| Persistence (tomorrow = today) | 47.7 | 0.717 |
| MLR | 41.5 | 0.786 |
| Random forest | **38.1** | **0.820** |

## 7. RQ2: predicting unhealthy days (notebook 06)

Test set: 2,963 days, 12.2% unhealthy (21.0% when the lockdown months are excluded). Thresholds were chosen on validation to maximise F2.

| Model | Test ROC-AUC | Test PR-AUC | Precision | Recall | F2 |
|---|---|---|---|---|---|
| Majority class | 0.500 | 0.122 | 0 | 0 | 0 |
| Persistence (today's label) | 0.963 | 0.811 | 0.736 | 0.740 | 0.740 |
| Today's AQI, tuned threshold (AQI ≥ 160) | 0.963 | 0.811 | 0.567 | 0.906 | 0.809 |
| Logistic regression (balanced) | 0.975 | 0.876 | 0.594 | 0.934 | 0.838 |
| LDA (shrinkage) | 0.971 | 0.859 | 0.550 | 0.936 | 0.821 |
| Decision tree (depth 4) | 0.960 | 0.823 | 0.657 | 0.906 | 0.842 |
| **Random forest** | **0.976** | 0.872 | 0.647 | 0.920 | **0.848** |
| KNN (k = 61) | 0.950 | 0.783 | 0.420 | 0.914 | 0.740 |
| ANN (64 hidden units) | 0.974 | **0.879** | 0.622 | 0.931 | 0.847 |

Accuracy is misleading here. The majority classifier has 87.8% accuracy and catches no unhealthy days.

**The honest comparison is with the tuned today's-AQI baseline,** not with plain persistence. Every model gets a threshold tuned on validation, so the baseline must get one too. Against that baseline, the best models add +0.013 ROC-AUC, +0.06 PR-AUC and +0.04 F2. That is a real but modest gain: most of the predictable signal is in today's AQI. Without the lockdown months the ranking is the same (ANN F2 0.895, RF 0.892, LogReg 0.889, baseline 0.858).

**Transitions** (healthy today, unhealthy tomorrow; validation and test pooled, 297 days): persistence catches 0%, the tuned baseline 71%, the random forest 75% and logistic regression 81%. The cost is false alarms on the day after an episode ends (91% for LogReg, 96% for RF). The models mainly learn "alarm when today is near the threshold and PM2.5 is rising".

**Interpretation.**
- The depth-3 tree reads: if today's AQI > 185, predict unhealthy. Otherwise predict unhealthy if today's PM2.5 > 84 µg/m³, especially when the 7-day mean is above 89.
- Logistic regression odds ratios per standard deviation: today's AQI 6.7, today's PM2.5 6.1, PM10 2.8, yesterday's AQI 2.5, CO 2.0. Yesterday's PM2.5 has 0.36: holding today fixed, a lower yesterday (rising pollution) raises the risk. These are correlated features, so read the ratios as a group.
- Regularisation hardly matters: validation AUC is flat across C, and the best value was 100, i.e. weak regularisation.
- The random forest's permutation importance agrees with this: PM2.5 today first, then today's AQI.
- **By city** (validation): recall is 0.94–1.00 in the high-pollution cities. It is lower in rarely unhealthy cities: Chennai 0.67 (12 positive days) and Jaipur 0.82.
- **Without Ahmedabad:** test AUC is unchanged (LogReg 0.975, RF 0.976).
- **Oversampling vs class weights:** almost identical (test AUC 0.975 for both).

## 8. RQ3: city profiles (notebook 07)

19 cities have at least one full year of data. The lockdown months are excluded. Each city is described by 16 features: the median and 90th percentile of PM2.5, NO, NO2, CO, SO2 and O3 (log scale); the shares of days Good/Satisfactory, Moderate and Poor-or-worse; and the winter/monsoon PM2.5 ratio. PM10, NOx, NH3, Benzene and Toluene are excluded because at least one city lacks them.

### 8.1 PCA and SVD
PCA via scikit-learn and via the SVD of the centred matrix give identical eigenvalues, loadings and scores (up to sign). PC1 explains 42%, PC2 18% and PC3 14%; four components explain 80%.
- **PC1 is the overall pollution level.** It loads positively on all PM2.5, NO, CO and SO2 features and on the Poor-or-worse share (0.37), and negatively on the Good/Satisfactory share (−0.33).
- **PC2 contrasts NO2 and O3** (urban traffic and photochemistry; loadings 0.37–0.50) with PM2.5, NO and the winter ratio. The eastern industrial towns score lowest on PC2.
- **PC3 picks out CO and SO2** (0.45, 0.37), an industrial-combustion axis.

### 8.2 Clusters
Silhouette is highest at k = 4 (0.38). k-means and Ward hierarchical clustering agree well (ARI 0.83).

| Cluster | Cities | Profile |
|---|---|---|
| Indo-Gangetic high pollution | Delhi, Gurugram, Lucknow, Patna, Kolkata | median PM2.5 86 µg/m³, 52% of days Poor or worse, winter PM2.5 4× monsoon |
| Cleaner southern and western cities | Thiruvananthapuram, Coimbatore, Bengaluru, Amaravati, Chennai, Hyderabad, Visakhapatnam, Jaipur, Amritsar | median PM2.5 38, 6% of days Poor or worse |
| Ahmedabad (alone) | Ahmedabad | median CO 17 mg/m³, SO2 48, 86% of days Poor or worse |
| Eastern industrial / mining towns | Talcher, Brajrajnagar, Jorapokhar, Guwahati | low NO2 and O3, PM2.5 55, strong winter peak (4.2×) |

**Geography.** All five cities in the first cluster are on the Indo-Gangetic Plain. Talcher and Brajrajnagar (Odisha) and Jorapokhar (Jharkhand) are coal and industrial areas. The misfits are Amritsar (Indo-Gangetic Plain, but in the cleaner cluster), Jaipur (northern, but clean in this data) and Guwahati (Northeast, grouped with the industrial towns because of its low NO2 and O3).

### 8.3 Stability
- Changing k: k = 3 merges the industrial towns into the Indo-Gangetic cluster (ARI 0.75); k = 5 isolates Jorapokhar and moves Amritsar into the industrial group (ARI 0.80).
- Bootstrap over days (50 resamples): mean ARI 0.96, minimum 0.51. The assignments are stable to sampling noise.
- **Without Ahmedabad:** ARI with the original clustering is 0.68. Ahmedabad takes up one of the four clusters, so without it k = 4 splits the clean cluster differently. The Indo-Gangetic cluster stays unchanged.
- With 19 cities this is a descriptive grouping, not a definitive typology.

## 9. Interpretation (synthesis)

1. **What drives AQI.** Fine particulate matter. PM2.5 is the binding pollutant on most unhealthy days in almost every city, and especially in winter and the post-monsoon season. All three analyses agree on this: RQ1 coefficients and importances, RQ2 features (today's PM2.5 and AQI) and RQ3 (PC1 is PM-led). CO looks important only because of Ahmedabad.
2. **Early warning.** A next-day warning is feasible. The simplest useful rule is "warn when today's AQI is at least 160", which catches 91% of unhealthy days in 2020. A model using PM2.5 trends raises recall to 92–93% with fewer false alarms (precision 0.65 vs 0.57). The gain is modest, so a deployed system should report its improvement over this baseline, not over plain persistence.
3. **Priorities.** The Indo-Gangetic cities (Delhi, Gurugram, Lucknow, Patna, Kolkata) from October to February: half of their days are Poor or worse, and their winter PM2.5 is four times the monsoon level. The eastern industrial towns are a second, distinct problem with different chemistry.
4. **Agreement between analyses.** The RQ3 clusters match where the RQ2 positives are concentrated: recall and positive rate are highest in the Indo-Gangetic cluster. The RQ1 binding-pollutant analysis explains the seasonality seen in EDA.

## 10. Limitations

- **Missing data and imputation.** Imputation error is about 0.73 standard deviations. Pollutants that a city never measured are filled entirely from other cities' patterns.
- **Uneven coverage.** Station coverage differs by city; 8 cities are excluded from modelling and 6 from profiling.
- **Circularity.** AQI is a deterministic function of the pollutants, so RQ1's same-day R² is partly circular. The binding-pollutant analysis and the tomorrow-AQI model address this.
- **No weather data.** Wind, rain, temperature and mixing height drive day-to-day changes, and their absence limits RQ2 most of all.
- **Lockdown.** The 2020 test year is atypical (positive rate 12% vs 25% in 2019). We report results with and without the lockdown months.
- **Suspect sensors.** Ahmedabad's CO, the exact zeros, and Talcher's O3 (+618% during the lockdown).
- **Few cities for clustering.** Only 19 cities, so the clusters are descriptive.
- **Final models trained on 2015–2018 only.** Refitting on 2015–2019 before testing would probably help, but would require refitting the imputer as well.

## 11. Future work
Add meteorological data, for example ERA5 reanalysis. Use time-series models (seasonal ARIMA, gradient boosting with longer lag windows). Forecast 2–3 days ahead. Model per station using `station_day.csv`. Recompute the official AQI from `city_hour.csv` to remove the daily approximation in the binding-pollutant analysis.

## 12. Conclusion
PM2.5 is the main driver of unhealthy air across Indian cities, above all on the Indo-Gangetic Plain in winter. Next-day unhealthy days can be flagged with high recall (test ROC-AUC 0.976), but most of that skill comes from today's AQI: the models' added value over a tuned baseline is real but small. Cities group into four interpretable profiles that are stable under resampling. One of those profiles, Ahmedabad, reflects a suspect CO sensor rather than real air quality.

## 13. Team roles and contributions [TEAM]

| Member | Role | Contributions |
|---|---|---|
| | Data cleaning and EDA | |
| | Regression, PCR, PCA, SVD | |
| | Classification and ROC analysis | |
| | Clustering and report writing | |

## References
- Vopani (Rohan Rao), *Air Quality Data in India (2015–2020)*, Kaggle.
- Central Pollution Control Board, *National Air Quality Index* (2014), and the CPCB data portal.
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12 (2011).
- Seabold & Perktold, *statsmodels*, Proc. SciPy (2010).
