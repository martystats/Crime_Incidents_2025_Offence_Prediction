# ================================================================
# Step 52: Import Libraries and Load Deployment Artifacts
# ================================================================

import os
import json
import joblib
import pandas as pd
import streamlit as st
from datetime import date, time


# ------------------------------------------------
# Configure Streamlit page
# ------------------------------------------------
st.set_page_config(
    page_title="Crime Incidents in 2025 Prediction",
    page_icon="🚔",
    layout="wide"
)


# ------------------------------------------------
# Get the folder containing app.py
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------
# Define deployment artifact paths
# ------------------------------------------------
MODEL_PATH = os.path.join(
    BASE_DIR,
    "final_logistic_regression_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "preprocessing_pipeline.pkl"
)

CLASS_LABELS_PATH = os.path.join(
    BASE_DIR,
    "class_labels.json"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "deployment_metadata.json"
)

INPUT_METADATA_PATH = os.path.join(
    BASE_DIR,
    "deployment_input_metadata.json"
)

PERFORMANCE_PATH = os.path.join(
    BASE_DIR,
    "model_performance.csv"
)

LOCATION_LOOKUP_PATH = os.path.join(
    BASE_DIR,
    "block_location_lookup.csv"
)

# ------------------------------------------------
# Load deployment artifacts
# ------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


@st.cache_data
def load_supporting_files():
    with open(
        CLASS_LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        class_labels = json.load(file)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        deployment_metadata = json.load(file)

    with open(
        INPUT_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        input_metadata = json.load(file)
    model_performance = pd.read_csv(
        PERFORMANCE_PATH
    )

    block_location_lookup = pd.read_csv(
        LOCATION_LOOKUP_PATH,
        dtype={
            "BLOCK": "string",
            "WARD": "string",
            "ANC": "string",
            "DISTRICT": "string",
            "PSA": "string",
            "NEIGHBORHOOD_CLUSTER": "string",
            "BLOCK_GROUP": "string",
            "CENSUS_TRACT": "string",
            "VOTING_PRECINCT": "string"
        }
    )
    
    block_location_lookup["BLOCK"] = (
        block_location_lookup["BLOCK"]
        .str.strip()
    )

    return (
    class_labels,
    deployment_metadata,
    input_metadata,
    model_performance,
    block_location_lookup
)


# ---------------------------------------------
# Load all files with error handling
# ---------------------------------------------
try:
    model, preprocessor = load_model_artifacts()

    (
    class_labels,
    deployment_metadata,
    input_metadata,
    model_performance,
    block_location_lookup
) = load_supporting_files()

    required_lookup_columns = [
        "BLOCK",
        "LATITUDE",
        "LONGITUDE",
        "WARD",
        "ANC",
        "DISTRICT",
        "PSA",
        "NEIGHBORHOOD_CLUSTER",
        "BLOCK_GROUP",
        "CENSUS_TRACT",
        "VOTING_PRECINCT"
    ]

    missing_lookup_columns = [
        column
        for column in required_lookup_columns
        if column not in block_location_lookup.columns
    ]

    if missing_lookup_columns:
        raise ValueError(
            "The location lookup file is missing required columns: "
            + ", ".join(missing_lookup_columns)
        )

    artifacts_loaded = True

except Exception as error:
    artifacts_loaded = False

    st.error(
        "The deployment artifacts could not be loaded."
    )

    st.exception(error)
    st.stop()


# ------------------------------------------------
# Basic application heading
# ------------------------------------------------
st.title("🚔 Crime Incidents in 2025 Prediction")

st.write(
    "This application uses a trained Logistic Regression model "
    "to predict the likely offence category of a crime incident."
)

# ----------------------------------------------------
# Access valid categorical values from metadata
# ----------------------------------------------------

categorical_values = input_metadata["categorical_values"]

def get_category_options(feature_name):
    """
    Return the valid training categories for a feature.
    """
    return categorical_values.get(feature_name, [])

# ============================================================
# Step 61: Replace Manual Location Inputs with Automatic Lookup
# ============================================================

st.divider()
st.subheader("Incident Information")

st.write(
    "Select the main incident details and the Incident Block. "
    "All related location information will be generated automatically."
)

# ------------------------------------------------------------
# Main incident inputs
# ------------------------------------------------------------
column_1, column_2 = st.columns(2)

with column_1:
    method = st.selectbox(
        "Incident Method",
        options=get_category_options("METHOD"),
        help="Select the method associated with the reported incident."
    )

with column_2:
    shift = st.selectbox(
        "Police Shift",
        options=get_category_options("SHIFT"),
        help="Select the police shift during which the incident was reported."
    )


# ------------------------------------------------------------
# Automatic block-location lookup
# ------------------------------------------------------------
st.subheader("Incident Location")

block_options = sorted(
    block_location_lookup["BLOCK"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

block = st.selectbox(
    "Incident Block",
    options=block_options,
    help=(
        "Search for and select the block where the incident occurred. "
        "All other location fields will be filled automatically."
    )
)

# Retrieve the complete location record for the selected block
selected_location = block_location_lookup.loc[
    block_location_lookup["BLOCK"].astype(str).str.strip() == str(block).strip()
].copy()

if selected_location.empty:
    st.error(
        "No matching location information was found for the selected block."
    )
    st.stop()

selected_location = selected_location.iloc[0]


# ------------------------------------------------------------
# Assign automatically generated location values
# ------------------------------------------------------------
latitude = float(selected_location["LATITUDE"])
longitude = float(selected_location["LONGITUDE"])

ward = str(selected_location["WARD"]).strip()
anc = str(selected_location["ANC"]).strip()
district = str(selected_location["DISTRICT"]).strip()
psa = str(selected_location["PSA"]).strip()

neighbourhood_cluster = str(
    selected_location["NEIGHBORHOOD_CLUSTER"]
).strip()

block_group = str(selected_location["BLOCK_GROUP"]).strip()
census_tract = str(selected_location["CENSUS_TRACT"]).strip()
voting_precinct = str(selected_location["VOTING_PRECINCT"]).strip()


# ------------------------------------------------------------
# Display automatic location summary
# ------------------------------------------------------------
with st.expander("View automatically generated location details"):

    summary_column_1, summary_column_2, summary_column_3 = st.columns(3)

    with summary_column_1:
        st.write(f"**Ward:** {ward}")
        st.write(f"**ANC:** {anc}")
        st.write(f"**District:** {district}")
        st.write(f"**Police Service Area:** {psa}")

    with summary_column_2:
        st.write(
            f"**Neighbourhood Cluster:** {neighbourhood_cluster}"
        )
        st.write(f"**Block Group:** {block_group}")
        st.write(f"**Census Tract:** {census_tract}")
        st.write(f"**Voting Precinct:** {voting_precinct}")

    with summary_column_3:
        st.write(f"**Latitude:** {latitude:.6f}")
        st.write(f"**Longitude:** {longitude:.6f}")

st.caption(
    "Location details are generated automatically from the selected Incident Block."
)


# ------------------------------------------------------------
# Incident date and time inputs
# ------------------------------------------------------------
st.subheader("Incident Date and Time")

date_column_1, date_column_2 = st.columns(2)

with date_column_1:

    start_date = st.date_input(
        "Incident Start Date",
        value=date(2025, 6, 30),
        help="Select the date on which the incident began."
    )

    start_time = st.time_input(
        "Incident Start Time",
        value=time(11, 0),
        help="Select the estimated time at which the incident began."
    )

with date_column_2:

    report_date = st.date_input(
        "Report Date",
        value=date(2025, 6, 30),
        help="Select the date on which the incident was reported."
    )

    report_time = st.time_input(
        "Report Time",
        value=time(12, 0),
        help="Select the time at which the incident was reported."
    )


# ================================================================
# Step 57: Construct the Complete Prediction Input Record
# ================================================================

import math


# ------------------------------------------------
# Helper function: assign time period
# ------------------------------------------------
def get_time_period(hour):
    if pd.isna(hour):
        return "Unknown"
    elif 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    else:
        return "Evening"


# ------------------------------------------------
# Helper function: assign season
# ------------------------------------------------
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


# ------------------------------------------------
# Helper function: calculate distance in kilometres
# ------------------------------------------------
def calculate_haversine_distance(
    latitude_1,
    longitude_1,
    latitude_2,
    longitude_2
):
    earth_radius_km = 6371.0

    latitude_1_radians = math.radians(latitude_1)
    latitude_2_radians = math.radians(latitude_2)

    latitude_difference = math.radians(latitude_2 - latitude_1)
    longitude_difference = math.radians(longitude_2 - longitude_1)

    haversine_value = (
        math.sin(latitude_difference / 2) ** 2
        + math.cos(latitude_1_radians)
        * math.cos(latitude_2_radians)
        * math.sin(longitude_difference / 2) ** 2
    )

    angular_distance = 2 * math.atan2(
        math.sqrt(haversine_value),
        math.sqrt(1 - haversine_value)
    )

    return earth_radius_km * angular_distance


# ------------------------------------------------
# Dataset geographical centre from feature engineering
# ------------------------------------------------
dataset_centre_latitude = 38.908618
dataset_centre_longitude = -77.013648


# ------------------------------------------------
# Combine selected dates and times
# ------------------------------------------------
report_datetime = pd.Timestamp.combine(
    report_date,
    report_time
)

start_datetime = pd.Timestamp.combine(
    start_date,
    start_time
)


# ------------------------------------------------
# Generate report date-time features
# ------------------------------------------------
report_year = report_datetime.year
report_month = report_datetime.month
report_day = report_datetime.day
report_hour = report_datetime.hour

report_day_of_week = report_datetime.day_name()
report_is_weekend = int(report_datetime.dayofweek >= 5)

report_time_period = get_time_period(report_hour)
report_season = get_season(report_month)


# ------------------------------------------------
# Generate incident-start date-time features
# ------------------------------------------------
start_year = start_datetime.year
start_month = start_datetime.month
start_day = start_datetime.day
start_hour = start_datetime.hour

start_day_of_week = start_datetime.day_name()
start_is_weekend = int(start_datetime.dayofweek >= 5)

start_time_period = get_time_period(start_hour)
start_season = get_season(start_month)


# ------------------------------------------------
# Generate spatial features
# ------------------------------------------------
latitude_distance_from_centre = (
    latitude - dataset_centre_latitude
)

longitude_distance_from_centre = (
    longitude - dataset_centre_longitude
)

distance_from_centre_km = calculate_haversine_distance(
    dataset_centre_latitude,
    dataset_centre_longitude,
    latitude,
    longitude
)

if latitude >= dataset_centre_latitude:
    latitude_direction = "North"
else:
    latitude_direction = "South"

if longitude >= dataset_centre_longitude:
    longitude_direction = "East"
else:
    longitude_direction = "West"

geographic_quadrant = (
    f"{latitude_direction}-{longitude_direction}"
)

latitude_rounded = round(latitude, 3)
longitude_rounded = round(longitude, 3)

location_grid = (
    f"{latitude_rounded}_{longitude_rounded}"
)


# ------------------------------------------------
# Generate contextual interaction features
# ------------------------------------------------
ward_shift = f"{ward} | {shift}"
district_shift = f"{district} | {shift}"

weekend_status = (
    "Weekend" if report_is_weekend == 1 else "Weekday"
)

weekend_shift = f"{weekend_status} | {shift}"


# ------------------------------------------------
# Build the complete prediction record
# ------------------------------------------------
prediction_record = {
    "BLOCK": block,
    "METHOD": method,
    "SHIFT": shift,
    "WARD": ward,
    "ANC": anc,
    "DISTRICT": district,
    "PSA": psa,
    "NEIGHBORHOOD_CLUSTER": neighbourhood_cluster,
    "BLOCK_GROUP": block_group,
    "CENSUS_TRACT": census_tract,
    "VOTING_PRECINCT": voting_precinct,

    "LATITUDE": latitude,
    "LONGITUDE": longitude,

    "REPORT_YEAR": report_year,
    "REPORT_MONTH": report_month,
    "REPORT_DAY": report_day,
    "REPORT_DAY_OF_WEEK": report_day_of_week,
    "REPORT_HOUR": report_hour,
    "REPORT_IS_WEEKEND": report_is_weekend,

    "START_YEAR": start_year,
    "START_MONTH": start_month,
    "START_DAY": start_day,
    "START_DAY_OF_WEEK": start_day_of_week,
    "START_HOUR": start_hour,
    "START_IS_WEEKEND": start_is_weekend,

    "LATITUDE_DISTANCE_FROM_CENTRE":
        latitude_distance_from_centre,

    "LONGITUDE_DISTANCE_FROM_CENTRE":
        longitude_distance_from_centre,

    "DISTANCE_FROM_CENTRE_KM":
        distance_from_centre_km,

    "GEOGRAPHIC_QUADRANT": geographic_quadrant,
    "LATITUDE_ROUNDED": latitude_rounded,
    "LONGITUDE_ROUNDED": longitude_rounded,
    "LOCATION_GRID": location_grid,

    "REPORT_TIME_PERIOD": report_time_period,
    "START_TIME_PERIOD": start_time_period,
    "REPORT_SEASON": report_season,
    "START_SEASON": start_season,

    "WARD_SHIFT": ward_shift,
    "DISTRICT_SHIFT": district_shift,
    "WEEKEND_SHIFT": weekend_shift
}


# ------------------------------------------------
# Convert the record into a one-row DataFrame
# ------------------------------------------------
prediction_dataframe = pd.DataFrame(
    [prediction_record]
)


# ------------------------------------------------
# Align columns with the original training order
# ------------------------------------------------
required_feature_order = input_metadata["feature_order"]

missing_features = [
    feature
    for feature in required_feature_order
    if feature not in prediction_dataframe.columns
]

extra_features = [
    feature
    for feature in prediction_dataframe.columns
    if feature not in required_feature_order
]

if missing_features:
    st.error(
        "The prediction record is missing required features: "
        + ", ".join(missing_features)
    )
    st.stop()

if extra_features:
    prediction_dataframe = prediction_dataframe.drop(
        columns=extra_features
    )

prediction_dataframe = prediction_dataframe[
    required_feature_order
]


# ------------------------------------------------
# Apply numerical and categorical data types
# ------------------------------------------------
numerical_features = deployment_metadata.get(
    "numerical_features",
    []
)

categorical_features = deployment_metadata.get(
    "categorical_features",
    []
)

for feature in numerical_features:
    if feature in prediction_dataframe.columns:
        prediction_dataframe[feature] = pd.to_numeric(
            prediction_dataframe[feature],
            errors="coerce"
        )

for feature in categorical_features:
    if feature in prediction_dataframe.columns:
        prediction_dataframe[feature] = (
            prediction_dataframe[feature]
            .astype(str)
        )


# ------------------------------------------------
# Validate the completed prediction record
# ------------------------------------------------
prediction_record_complete = (
    prediction_dataframe.shape[1]
    == len(required_feature_order)
)

prediction_missing_values = int(
    prediction_dataframe.isna().sum().sum()
)


    # ============================================================
# Step 58: Generate Crime Offence Prediction
# ============================================================

st.divider()
st.subheader("Offence Prediction")

st.write(
    "Click the button below to process the incident information "
    "and predict the most likely offence category."
)

predict_button = st.button(
    "Predict Offence Category",
    type="secondary",
    use_container_width=True
)

if predict_button:

    if not prediction_record_complete:
        st.error(
            "The prediction record does not contain all required features."
        )

    elif prediction_missing_values > 0:
        st.error(
            "The prediction record contains missing values. "
            "Please check the incident information."
        )

    else:
        try:
            processed_prediction_record = preprocessor.transform(
                prediction_dataframe
            )

            predicted_offence = model.predict(
                processed_prediction_record
            )[0]

            prediction_probabilities = model.predict_proba(
                processed_prediction_record
            )[0]

            probability_class_labels = list(model.classes_)

            probability_table = pd.DataFrame(
                {
                    "Offence Category": probability_class_labels,
                    "Probability": prediction_probabilities
                }
            )

            probability_table["Confidence (%)"] = (
                probability_table["Probability"] * 100
            ).round(2)

            probability_table = (
                probability_table
                .sort_values(
                    by="Probability",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            predicted_confidence = float(
                probability_table.loc[
                    probability_table["Offence Category"]
                    == predicted_offence,
                    "Confidence (%)"
                ].iloc[0]
            )

            st.success("Prediction completed successfully.")

            st.markdown("### Prediction Summary")

            summary_column_1, summary_column_2 = st.columns([2.5, 1])

            with summary_column_1:
                st.info(
                    f"**Predicted Offence:**  \n"
                    f"### {predicted_offence}"
                )

            with summary_column_2:
                st.metric(
                    label="Model Confidence",
                    value=f"{predicted_confidence:.2f}%"
                )

            st.subheader("Top 3 Prediction Probabilities")

            top_three_probabilities = probability_table.head(3)

            st.dataframe(
                top_three_probabilities[
                    ["Offence Category", "Confidence (%)"]
                ],
                use_container_width=True,
                hide_index=True
            )

            with st.expander("View all prediction probabilities"):
                st.dataframe(
                    probability_table[
                        ["Offence Category", "Confidence (%)"]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "The confidence percentages represent the probabilities "
                    "assigned by the Logistic Regression model."
                )

        except Exception as prediction_error:
            st.error(
                "The prediction could not be completed."
            )

            with st.expander("View technical error details"):
                st.exception(prediction_error)