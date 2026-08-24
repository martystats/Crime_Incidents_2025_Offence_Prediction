# Crime Incidents in 2025 – Offence Prediction

**Project Developer:** Martin Jude
**GitHub:** martystats  
**Project Type:** End-to-End Machine Learning Classification & Streamlit Deployment  
**Year:** 2026

A multiclass machine-learning project developed to predict the most likely **OFFENSE** category for a crime incident using incident, temporal and location-related information.

The project covers the complete machine-learning workflow, including data preparation, exploratory data analysis, feature engineering, model development, validation, reproducibility testing and deployment through an interactive Streamlit application.

The final balanced Logistic Regression model is deployed with automatic location-feature generation and class-probability estimates.

---

## Project Overview

This project analyses public crime incident records from 2025 and develops a multiclass classification model for predicting offence categories.

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
- Notebook reproducibility testing
- Streamlit application development
- Automatic location feature generation

---

## Data Source

The dataset used in this project is **Crime Incidents in 2025**, published by the District of Columbia Metropolitan Police Department (MPD).

- **Dataset:** Crime Incidents in 2025
- **Publisher:** District of Columbia Metropolitan Police Department (MPD)
- **Source:** https://opendata.dc.gov/datasets/DCGIS::crime-incidents-in-2025
- **Accessed:** 15 August 2026
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset contains a subset of locations and attributes of crime incidents reported through the Metropolitan Police Department's crime reporting system.

---

## Raw Data Placement

The raw dataset is kept separate from generated and cleaned data so that notebook execution cannot overwrite the original downloaded CSV.

After downloading the dataset, place the original CSV at:

```text
Data/
└── raw/
    └── Crime_Incidents_in_2025.csv
```

The notebook uses separate paths for the raw and cleaned datasets:

```python
RAW_DATA_PATH = Path(
    "Data/raw/Crime_Incidents_in_2025.csv"
)

CLEAN_DATA_PATH = Path(
    "Data/processed/crime_incidents_2025_clean.csv"
)
```

The original raw CSV is therefore preserved throughout the workflow.

Cleaned data is written to and reloaded through `CLEAN_DATA_PATH`.

This separation also prevents filename conflicts on case-insensitive operating systems such as Windows.

---

## Prediction Target

The model predicts one of the following eight offence categories:

- ASSAULT W/DANGEROUS WEAPON
- BURGLARY
- HOMICIDE
- MOTOR VEHICLE THEFT
- ROBBERY
- SEX ABUSE
- THEFT F/AUTO
- THEFT/OTHER

The original **ARSON** category was removed because it contained only four records, which was insufficient for reliable model training and evaluation.

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

Categorical selection included pandas `object`, `category` and `string` data types. This ensured that categorical predictors were handled consistently by the preprocessing pipeline.

---

## Feature Engineering

The modelling workflow created and used features related to:

- Incident method
- Police shift
- Incident block
- Ward
- Advisory Neighborhood Commission
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
- Additional date-derived predictors
- Additional spatial predictors
- Distance-from-centre features

The distance-from-centre features are calculated using **absolute values**, maintaining consistency between the training notebook and the deployed Streamlit application.

The fitted preprocessing pipeline transforms the **39 raw predictors into 11,712 processed model features**.

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

This gives additional importance to less frequent offence classes during model training.

---

## Final Test Performance

The selected model was evaluated once on the untouched test dataset containing **2,408 records**.

| Metric | Test Score |
|---|---:|
| Accuracy | 0.4671 |
| Balanced Accuracy | 0.4227 |
| Macro Precision | 0.3622 |
| Macro Recall | 0.4227 |
| Macro F1 Score | 0.3793 |

The test results were close to, and slightly stronger than, the validation results. This supports reasonable generalisation to unseen test observations.

Balanced Accuracy of approximately **42.27%** is above the eight-class chance level of **12.5%**.

However, performance varies between offence categories because the target classes remain imbalanced.

---

## Artifact Reload Validation

The saved model and preprocessing artifacts were independently reloaded and tested to confirm that they reproduce the original model behaviour.

| Validation Measurement | Result |
|---|---:|
| Raw test shape | 2,411 × 39 |
| Processed test shape | 2,411 × 11,741 |
| Processed feature names | 11,741 |
| Model coefficient features | 11,741 |
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
- View the **Model Probability (uncalibrated)**
- View the top three class probabilities
- Expand the complete probability table when required

The application validates the incident and report timestamps so that the **report date/time cannot be earlier than the incident start date/time**.

Selecting an incident block automatically supplies:

- Ward
- Advisory Neighborhood Commission
- Police district
- Police Service Area
- Neighbourhood cluster
- Block group
- Census tract
- Voting precinct
- Latitude
- Longitude
- Other required location-derived values

The application constructs the complete **39-feature input record internally** and aligns it with the exact feature order used during training.

High-cardinality predictors such as `BLOCK` and `LOCATION_GRID` are supplied through the automatic location process instead of large manual dropdown menus.

---

## Application Interface

The final interface includes:

- A two-column incident-input layout
- Automatic location lookup
- Incident-start details before report details
- A full-width prediction button
- A prediction summary
- Model Probability (uncalibrated) display
- Top-three prediction probabilities
- An expandable table containing all class probabilities
- Hidden technical debugging sections
- Educational-use warning

---

## Deployment Architecture

The application follows this prediction process:

1. The user supplies the visible incident inputs.
2. The selected incident block is matched with the location lookup dataset.
3. Required location and spatial values are populated automatically.
4. Distance-from-centre features are calculated using absolute values.
5. A complete 39-predictor record is constructed.
6. `deployment_input_metadata.json` supplies the required raw-feature order.
7. The fitted preprocessing pipeline transforms the record.
8. The balanced Logistic Regression model generates the prediction.
9. The predicted offence, uncalibrated model probability and class probabilities are displayed.

---

## Project Files

```text
crime_incidents_streamlit_app/
│
├── app.py
├── Crime_incidents_in_2025_Workflow.ipynb
├── final_logistic_regression_model.pkl
├── preprocessing_pipeline.pkl
├── class_labels.json
├── model_performance.csv
├── deployment_metadata.json
├── deployment_input_metadata.json
├── block_location_lookup.csv
├── requirements.txt
├── requirements-notebook.txt
├── README.md
│
└── Data/
    ├── raw/
    │   └── Crime_Incidents_in_2025.csv
    │
    └── processed/
        └── crime_incidents_2025_clean.csv
```

---

## File Purposes

| File | Purpose |
|---|---|
| `app.py` | Streamlit prediction application |
| `Crime_incidents_in_2025_Workflow.ipynb` | Complete data-analysis and machine-learning notebook |
| `final_logistic_regression_model.pkl` | Final balanced Logistic Regression model |
| `preprocessing_pipeline.pkl` | Fitted preprocessing pipeline |
| `class_labels.json` | Ordered target-class labels |
| `model_performance.csv` | Final untouched-test performance |
| `deployment_metadata.json` | Model structure, metrics and deployment information |
| `deployment_input_metadata.json` | Raw-feature order, categorical values and numerical ranges |
| `block_location_lookup.csv` | Automatic block-to-location lookup data |
| `requirements.txt` | Python packages required for the Streamlit application |
| `requirements-notebook.txt` | Python packages required to reproduce the notebook workflow |
| `README.md` | Project and deployment documentation |
| `Data/raw/Crime_Incidents_in_2025.csv` | Original downloaded raw dataset |
| `Data/processed/crime_incidents_2025_clean.csv` | Cleaned dataset generated by the notebook |

---

## Notebook Reproducibility

To reproduce the complete analysis, first download the original **Crime Incidents in 2025** CSV dataset.

Place the downloaded CSV at:

```text
Data/raw/Crime_Incidents_in_2025.csv
```

Install the notebook dependencies:

```bash
pip install -r requirements-notebook.txt
```

Then open:

```text
Crime_incidents_in_2025_Workflow.ipynb
```

Restart the notebook kernel and run all cells sequentially from top to bottom.

The raw and processed datasets use deliberately different directories and filenames:

```text
Raw:
Data/raw/Crime_Incidents_in_2025.csv

Processed:
Data/processed/crime_incidents_2025_clean.csv
```

This prevents the cleaned dataset from overwriting the original raw dataset, particularly on case-insensitive operating systems such as Windows.

A fresh-kernel, top-to-bottom execution is used to verify that the workflow is reproducible from the original raw data.

---

## Running the Application Locally

Open a terminal inside the project folder and install the application dependencies:

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
- The displayed model probabilities are **uncalibrated** and should not be interpreted as calibrated confidence estimates.
- Predictions represent statistical estimates and must not be interpreted as confirmed crime classifications.
- The application is intended for **educational, analytical and portfolio demonstration purposes only**.
- The model must not be used for operational policing, legal decisions, individual risk assessment, resource allocation or other high-stakes decisions.

---

## Reproducibility Status

The final notebook was tested using a fresh-kernel, top-to-bottom execution after separating the raw and processed dataset paths.

The workflow completed without errors, while the final model performance and artifact-reload validation results remained consistent.

The saved model, preprocessing pipeline, metadata and deployment application therefore form a reproducible end-to-end machine-learning workflow.

---

## Project Status

**Completed**

This project was developed, tested and documented by **Martin Jamed** as an end-to-end machine-learning classification project.

The work covers:

- Raw data preparation
- Exploratory data analysis
- Feature engineering
- Machine-learning model comparison
- Class-imbalance handling
- Model evaluation
- Robustness and validation testing
- Reproducibility testing
- Model artifact validation
- Streamlit application development
- Deployment preparation
- Technical documentation

The final model, preprocessing pipeline and deployment artifacts were validated successfully.

The completed project is suitable for **GitHub publication and data science / machine-learning portfolio presentation**.

---

## Author

**Martin Ude**  
Data Analysis & Machine Learning  
GitHub: **martystats**

---

© 2026 Martin Ude