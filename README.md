# Crime Incidents in 2025 – Offence Prediction

A machine learning classification project that predicts the most likely offence category for a crime incident using incident, location, date and time information.

The final model is deployed through an interactive Streamlit application with automatic location lookup, probability estimates and a clean prediction summary.

---

## Project Overview

This project analyses crime incident records from 2025 and develops a multiclass classification model for predicting the `OFFENSE` category.

The workflow covers:

- Data cleaning and type correction
- Exploratory data analysis
- Feature engineering
- Rare-class handling
- Model training and comparison
- Final model evaluation
- Deployment artifact creation
- Streamlit application development
- Automatic location feature generation

The deployed model is a Logistic Regression classifier.

---

## Prediction Target

The model predicts one of the following offence categories:

- Assault with a dangerous weapon
- Burglary
- Homicide
- Motor vehicle theft
- Robbery
- Sex abuse
- Theft from auto
- Theft/other

The original `ARSON` category was removed because it contained too few records for reliable model training and evaluation.

---

## Dataset Preparation

The final modelling dataset contains:

- 24,075 records
- 39 predictor features
- 8 target classes
- No missing predictor values
- No missing target values

Identifier, categorical, continuous, spatial and date-derived variables were reviewed and prepared before modelling.

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
- Report date and time
- Incident start date and time
- Month
- Day
- Hour
- Day of week
- Weekend indicator
- Time period
- Season
- Additional date and location-derived predictors

---

## Models Evaluated

The following classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Logistic Regression was selected as the final deployment model after validation and test evaluation.

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
- Expand the full probability table when needed

Selecting an incident block automatically populates:

- Ward
- ANC
- District
- Police Service Area
- Neighbourhood cluster
- Block group
- Census tract
- Voting precinct
- Latitude
- Longitude

The application constructs the complete 39-feature model input record internally and aligns it with the feature order used during training.

---

## Application Interface

The final interface includes:

- A two-column incident input layout
- Automatic location lookup
- Incident start details before report details
- A neutral full-width prediction button
- A polished prediction summary
- Model confidence display
- Top three prediction probabilities
- An expandable table for all class probabilities
- Hidden technical debugging sections

---

## Project Files

```text
crime_incidents_streamlit_app/
│
├── app.py
├── final_logistic_regression_model.pkl
├── preprocessing_pipeline.pkl
├── deployment_metadata.json
├── class_labels.json
├── model_performance.csv
├── block_location_lookup.csv
└── README.md

---

## Author

**Martin Ude**

Data Analyst and Machine Learning Practitioner

GitHub: `martystats`