# Crime Incidents in 2025 – Offence Prediction

A multiclass machine-learning project that predicts the most likely `OFFENSE` category for a crime incident using incident, location, date and time information.

The final balanced Logistic Regression model is deployed through an interactive Streamlit application with automatic location lookup, confidence scores and class-probability estimates.

---

## Project Overview

This project analyses public crime incident records from 2025 and develops a multiclass classification model for predicting eight offence categories.

The workflow includes:

- Data cleaning and type correction
- Exploratory data analysis
- Feature engineering
- Rare-class handling
- Train, validation and test splitting
- Preprocessing pipeline construction
- Model training and comparison
- Cross-validation and robustness checks
- Grouped and temporal validation
- Final untouched-test evaluation
- Artifact saving and reload validation
- Streamlit application development
- Automatic location feature generation

---

## Prediction Target

The model predicts one of the following eight offence categories:

- `ASSAULT W/DANGEROUS WEAPON`
- `BURGLARY`
- `HOMICIDE`
- `MOTOR VEHICLE THEFT`
- `ROBBERY`
- `SEX ABUSE`
- `THEFT F/AUTO`
- `THEFT/OTHER`

The original `ARSON` category was removed because it contained only four records, which was insufficient for reliable model training and evaluation.

---

## Dataset Preparation

The final modelling dataset contains:

| Measurement | Result |
|---|---:|
| Records | 24,075 |
| Raw predictor features | 39 |
| Numerical predictors | 17 |
| Categorical predictors | 22 |
| Target classes | 8 |
| Missing predictor values | 0 |
| Missing target values | 0 |

Identifier, categorical, continuous, spatial and date-derived variables were reviewed and prepared before modelling.

Categorical selection included pandas `object`, `category` and `string` data types. This ensured that all intended categorical predictors were included in the preprocessing pipeline.

---

## Feature Engineering

The modelling workflow created and used features related to:

- Incident method
- Police shift
- Incident block
- Ward
- Advisory Neighbourhood Commission
- Police district
- Police Service Area
- Neighbourhood cluster
- Block group
- Census tract
- Voting precinct
- Latitude and longitude
- Location grid
- Report date and time
- Incident start date and time
- Month
- Day
- Hour
- Day of week
- Weekend indicator
- Time period
- Season
- Additional date-derived and spatial predictors

The fitted preprocessing pipeline transforms the 39 raw predictors into 11,712 processed model features.

---

## Models Evaluated

The following classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Balanced Logistic Regression was selected as the final model after validation, robustness analysis and untouched-test evaluation.

The model uses:

```python
class_weight="balanced"
```

This gives additional importance to less frequent offence classes during training.

---

## Final Test Performance

The selected model was evaluated once on the untouched test dataset containing 2,408 records.

| Metric | Test Score |
|---|---:|
| Accuracy | 0.5071 |
| Balanced Accuracy | 0.4227 |
| Macro Precision | 0.3622 |
| Macro Recall | 0.4227 |
| Macro F1 Score | 0.3793 |

The test results were close to, and slightly stronger than, the validation results. This supports reasonable generalisation to unseen randomly sampled records.

Balanced Accuracy of approximately 42.27% is above the eight-class chance level of 12.50%. However, performance varies substantially between offence categories because the target remains imbalanced.

---

## Artifact Reload Validation

The saved model and preprocessing artifacts were independently reloaded and tested.

| Validation Measurement | Result |
|---|---:|
| Raw test shape | 2,408 × 39 |
| Processed test shape | 2,408 × 11,712 |
| Processed feature names | 11,712 |
| Model coefficient features | 11,712 |
| Exact prediction match | True |
| Prediction agreement | 100% |
| Metric differences | 0.0 |

All eleven validation checks passed, including:

- Required artifact availability
- Raw-feature ordering
- Processed-feature names
- Processed-matrix shape and values
- Exact prediction reproduction
- Class-label ordering
- Performance-metric reproduction
- Model-feature structure
- Balanced class weighting
- Deployment metadata consistency

This confirms that the saved artifacts reproduce the original model behaviour exactly.

---

## Streamlit Application

The Streamlit application allows a user to:

- Select the incident method
- Select the police shift
- Select an incident block
- Enter incident start date and time
- Enter report date and time
- Generate location details automatically
- Produce an offence prediction
- View the model confidence
- View the top three class probabilities
- Expand the complete probability table when required

Selecting an incident block automatically supplies:

- Ward
- Advisory Neighbourhood Commission
- Police district
- Police Service Area
- Neighbourhood cluster
- Block group
- Census tract
- Voting precinct
- Latitude
- Longitude
- Other required location-derived values

The application constructs the complete 39-feature input record internally and aligns it with the exact feature order used during model training.

High-cardinality predictors such as `BLOCK` and `LOCATION_GRID` are supplied through the automatic location process instead of being displayed as very large manual dropdown menus.

---

## Application Interface

The final interface includes:

- A two-column incident-input layout
- Automatic location lookup
- Incident-start details before report details
- A full-width prediction button
- A polished prediction summary
- Model-confidence display
- Top-three prediction probabilities
- An expandable table containing all class probabilities
- Hidden technical debugging sections

---

## Deployment Architecture

The application follows this prediction process:

1. The user supplies the visible incident inputs.
2. The selected block is matched with the location lookup dataset.
3. Required location and spatial values are populated automatically.
4. A complete 39-predictor record is constructed.
5. `deployment_input_metadata.json` supplies the required raw-feature order.
6. The fitted preprocessing pipeline transforms the record.
7. The balanced Logistic Regression model generates the prediction.
8. The predicted offence, confidence and class probabilities are displayed.

---

## Project Files

```text
crime_incidents_streamlit_app/
│
├── app.py
├── final_logistic_regression_model.pkl
├── preprocessing_pipeline.pkl
├── class_labels.json
├── model_performance.csv
├── deployment_metadata.json
├── deployment_input_metadata.json
├── block_location_lookup.csv
├── requirements.txt
└── README.md
```

### File Purposes

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `final_logistic_regression_model.pkl` | Corrected balanced Logistic Regression model |
| `preprocessing_pipeline.pkl` | Fitted preprocessing pipeline |
| `class_labels.json` | Ordered target-class labels |
| `model_performance.csv` | Final untouched-test performance |
| `deployment_metadata.json` | Model structure, metrics and deployment status |
| `deployment_input_metadata.json` | Raw-feature order, categorical values and numerical ranges |
| `block_location_lookup.csv` | Automatic block-to-location lookup data |
| `requirements.txt` | Required Python packages |
| `README.md` | Project and deployment documentation |

---

## Running the Application Locally

Open a terminal inside the deployment folder and install the required packages:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local address that can be opened in a web browser.

---

## Limitations and Responsible Use

- The target classes are substantially imbalanced.
- Model performance differs across offence categories.
- Performance is particularly weak for rare classes such as `SEX ABUSE`.
- Grouped validation indicates that performance may decrease for unfamiliar locations.
- Temporal validation indicates that performance may change as offence patterns and class distributions change.
- Predictions represent statistical estimates and must not be interpreted as confirmed crime classifications.
- The model must not be used for operational policing, legal decisions or other high-stakes decisions.

This project is intended solely for education, portfolio demonstration and analytical research.

---

## Author

**Martin Jude**

Data Analyst and Machine Learning Practitioner

GitHub: **martystats**