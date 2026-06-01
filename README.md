# DigiCow Farmer Adoption Prediction 🐂
## About DigiCow

<img width="612" height="408" alt="image" src="https://github.com/user-attachments/assets/5f1ab866-55f2-47d9-a424-7ace33c30b6b" />



DigiCow Africa LTD is an award-winning Kenyan agritech company working with over 200,000 smallholder farmers through mobile-based training and digital extension services. Named Kenya's most innovative agritech by the World Bank in 2019, DigiCow delivers audio-based training content on animal health, crop management and record-keeping directly to farmers' phones, replacing the traditional model of infrequent extension officer visits.

Despite reaching farmers at scale, adoption rates after training sessions remain low at approximately 1-2%. This project builds interpretable machine learning models to predict which farmers are most likely to adopt the practices  taught after training. This enables DigiCow to prioritise follow-up support and design more effective extension strategies.

## The Challenge
Agricultural training programs are widely used to improve productivity among smallholder farmers. However, attending training does not necessarily translate to adoption of recommended practices. Understanding which farmers are most likely to adopt new techniques can help organizations target resources more effectively and improve program design.

In this project, I developed predictive models to identify factors associated with the adoption of dairy farming practices among Kenyan farmers participating in the DigiCow training program. Rather than focusing solely on model performance, I was interested in understanding which factors consistently influenced adoption and how those insights could support more effective agricultural extension strategies.

Adoption rates after agricultural training are very low and uneven; approximately 1-2% within any prediction window. Identifying which farmers are most likely to adopt allows DigiCow to prioritise follow-up support and design more effective training programmes.

## Data
| File | Description |
|---|---|
| Train.csv | 13,536 labelled training records |
| Test.csv | 5,621 unlabelled records for prediction |
| Prior.csv | 44,882 historical records used for feature engineering |

## Approach

The project followed a structured machine learning workflow:

- Data cleaning and validation
- Feature engineering
- Training-validation split to prevent information leakage
- Logistic Regression as an interpretable baseline model
- Decision Tree modelling to capture potential non-linear relationships
- Classifier Chain modelling to account for dependencies between adoption outcomes
- Model evaluation using AUC-ROC

Special attention was given to class imbalance and reproducibility throughout the modelling process.

### Feature Engineering
Six features were constructed from the available data:
- `belong_to_cooperative` - binary flag, direct from training record
- `is_ussd_registered` - binary flag, recoded from registration method
- `trainer_rate_smoothed` - Bayesian smoothed historical adoption rate per trainer
- `county_rate_smoothed` - Bayesian smoothed historical adoption rate per county
- `topic_rate_smoothed` - mean smoothed adoption rate across session topics
- `n_topics` - number of unique topics covered per session

Smoothed rates were computed exclusively from Prior.csv to prevent data leakage. Bayesian smoothing pulls estimates from small groups toward the global mean, preventing unreliable rates from rare trainers or counties from dominating predictions.

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

### Limitations and Considerations
It is worth noting that several engineered variables were derived from historical adoption behaviour. While these features improved predictive performance, they may not generalize perfectly to new trainers, counties, or training topics with limited historical data.

In addition, adoption behaviour is influenced by external factors such as market conditions, weather, and resource availability, which were not captured in the dataset.
## Leaderboard Results
| Model | Public AUC | Notes |
|---|---|---|
| Logistic Regression (this notebook) | 0.664 | Interpretable-6 features |
| LightGBM + feature engineering | 0.783 | Complex-83 features |

## Key Findings

- Trainer performance emerged as the strongest predictor of adoption. Farmers assigned to the highest-performing trainers adopted at substantially higher rates than those assigned to lower-performing trainers, suggesting that extension delivery quality plays a critical role in behaviour change.

- Health-related topics such as vaccination and deworming consistently generated higher adoption rates than other content areas, indicating that topic relevance may influence farmer engagement.

- Farmers who registered through USSD channels adopted at higher rates, suggesting that digital engagement may serve as a useful proxy for motivation or programme readiness.


## Author's Note

This was actually my second attempt at the DigiCow challenge.

My first submission achieved a better leaderboard score, but when I revisited the project, I wanted to understand the problem more deeply rather than simply maximise performance. I stripped the model back, engineered a small number of features from first principles, and focused on building something I could explain.

What surprised me most was that trainer and topic effects consistently mattered more than I expected. I went into the project assuming adoption would be driven mainly by farmer characteristics. Instead, the models suggested that how training is delivered and what is being taught may be just as important.

The project also reinforced something I keep encountering in data work: good performance does not always mean good understanding. Although this version scored lower on the leaderboard, I came away with a much clearer picture of the problem and a stronger appreciation for the trade-offs between predictive accuracy, interpretability, and practical decision-making.

I intend to continue improving the model by incorporating additional features, exploring more advanced algorithms, and testing whether the findings remain consistent across different modelling approaches.


## Tools
Python 3.13 · pandas · numpy · scikit-learn · matplotlib · seaborn

