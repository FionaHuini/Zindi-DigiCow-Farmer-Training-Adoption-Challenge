# DigiCow Farmer Adoption Prediction

## Overview
This project predicts the probability that a smallholder farmer adopts an agricultural practice within 7, 90 and 120 days of attending a 
training session. The analysis was developed as part of the DigiCow Farmer Adoption Prediction challenge on Zindi Africa.

The focus throughout is on interpretable models whose predictions and feature importance scores can be explained to a non-technical audience.

## Problem
Adoption rates after agricultural training are very low and uneven; approximately 1-2% within any prediction window. Identifying which farmers are most likely to adopt allows DigiCow to prioritise follow-up 
support and design more effective training programmes.

## Data
| File | Description |
|---|---|
| Train.csv | 13,536 labelled training records |
| Test.csv | 5,621 unlabelled records for prediction |
| Prior.csv | 44,882 historical records used for feature engineering |

## Approach

### Feature Engineering
Six features were constructed from the available data:
- `belong_to_cooperative` — binary flag, direct from training record
- `is_ussd_registered` — binary flag, recoded from registration method
- `trainer_rate_smoothed` — Bayesian smoothed historical adoption rate per trainer
- `county_rate_smoothed` — Bayesian smoothed historical adoption rate per county
- `topic_rate_smoothed` — mean smoothed adoption rate across session topics
- `n_topics` — number of unique topics covered per session

Smoothed rates were computed exclusively from Prior.csv to prevent data leakage. Bayesian smoothing pulls estimates from small groups 
toward the global mean, preventing unreliable rates from rare trainers or counties from dominating predictions.

### Models
Three models were built and compared:

| Model | Key strength |
|---|---|
| Logistic Regression | Interpretable odds ratios per feature |
| Decision Tree | Visual decision rules, captures non-linear patterns |
| Classifier Chain | Exploits nested target structure for consistent predictions |

### Results

| Target | Logistic Regression | Decision Tree | Classifier Chain |
|---|---|---|---|
| 7-day | 0.914 | 0.873 | 0.913 |
| 90-day | 0.860 | 0.880 | 0.883 |
| 120-day | 0.860 | 0.883 | 0.860 |

All models substantially outperform random guessing (AUC=0.5).

## Leaderboard Results
| Model | Public AUC | Notes |
|---|---|---|
| Logistic Regression (this notebook) | 0.664 | Interpretable-6 features |
| LightGBM + feature engineering | 0.783 | Complex-83 features |

## Key Findings
- Trainer quality is the strongest predictor; the best trainer achieves 5.31% adoption vs 0.14% for the worst
- Health-focused topics (poultry health, deworming, vaccination) drive the highest adoption rates; up to 24% for some topics
- USSD-registered farmers adopt at 3x the rate of manually registered farmers
- Classifier chain improves 90-day prediction by conditioning on the 120-day estimate

## Tools
Python 3.13 · pandas · numpy · scikit-learn · matplotlib · seaborn

