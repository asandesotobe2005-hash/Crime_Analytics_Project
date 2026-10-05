import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Crime Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("South Africa Crime Analytics Dashboard")

st.write(
    "An interactive dashboard for exploring crime patterns, "
    "police station information and crime statistics."
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

station_df = pd.read_csv("station_clean.csv")
crime_df = pd.read_csv("crime_clean.csv")

crime_df["month"] = pd.to_datetime(crime_df["month"])

crime_df["year"] = crime_df["month"].dt.year


# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------

st.sidebar.header("Filters")

# Province filter
provinces = sorted(station_df["province"].unique())

selected_province = st.sidebar.selectbox(
    "Select Province",
    ["All Provinces"] + provinces
)

# Crime category filter
categories = sorted(crime_df["category"].unique())

selected_category = st.sidebar.selectbox(
    "Select Crime Category",
    ["All Categories"] + categories
)

# Year filter
years = sorted(crime_df["year"].unique())

selected_year = st.sidebar.selectbox(
    "Select Year",
    ["All Years"] + years
)


# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------

filtered_crime = crime_df.copy()

if selected_province != "All Provinces":

    province_stations = station_df[
        station_df["province"] == selected_province
    ]["station_id"]

    filtered_crime = filtered_crime[
        filtered_crime["station_id"].isin(province_stations)
    ]


if selected_category != "All Categories":

    filtered_crime = filtered_crime[
        filtered_crime["category"] == selected_category
    ]


if selected_year != "All Years":

    filtered_crime = filtered_crime[
        filtered_crime["year"] == selected_year
    ]


# ---------------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Police Stations",
        station_df["station_id"].nunique()
    )

with col2:

    st.metric(
        "Filtered Crime Records",
        len(filtered_crime)
    )

with col3:

    st.metric(
        "Crime Categories",
        filtered_crime["category"].nunique()
    )


# ---------------------------------------------------------
# EDA SECTION
# ---------------------------------------------------------

st.header("Exploratory Data Analysis")

st.write(
    "The following visualisations show crime patterns based on "
    "the selected filters."
)


# ---------------------------------------------------------
# CRIME BY CATEGORY
# ---------------------------------------------------------

st.subheader("Crime by Category")

category_crime = (
    filtered_crime
    .groupby("category")["incidents"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(category_crime)


# ---------------------------------------------------------
# CRIME BY YEAR
# ---------------------------------------------------------

st.subheader("Crime by Year")

yearly_crime = (
    filtered_crime
    .groupby("year")["incidents"]
    .sum()
    .sort_index()
)

st.line_chart(yearly_crime)


# ---------------------------------------------------------
# CRIME BY PROVINCE
# ---------------------------------------------------------

st.subheader("Crime by Province")

province_crime = (
    filtered_crime
    .merge(
        station_df[["station_id", "province"]],
        on="station_id",
        how="left"
    )
    .groupby("province")["incidents"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(province_crime)


# ---------------------------------------------------------
# CRIME BY URBAN TYPE
# ---------------------------------------------------------

st.subheader("Crime by Urban Type")

urban_crime = (
    filtered_crime
    .merge(
        station_df[["station_id", "urban_type"]],
        on="station_id",
        how="left"
    )
    .groupby("urban_type")["incidents"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(urban_crime)


# ---------------------------------------------------------
# MONTHLY CRIME TREND
# ---------------------------------------------------------

st.subheader("Monthly Crime Trend")

monthly_crime = (
    filtered_crime
    .groupby("month")["incidents"]
    .sum()
    .sort_index()
)

st.line_chart(monthly_crime)


# ---------------------------------------------------------
# FILTERED DATA
# ---------------------------------------------------------

st.header("Filtered Crime Data")

st.dataframe(
    filtered_crime.head(100),
    use_container_width=True
)
# ==========================================================
# MACHINE LEARNING CLASSIFICATION
# ==========================================================

st.header("Machine Learning Classification")

st.write(
    "Machine learning models are used to classify police stations "
    "as either crime hotspots or non-hotspots based on demographic, "
    "geographic and socioeconomic characteristics."
)

# ----------------------------------------------------------
# Create station-level crime dataset
# ----------------------------------------------------------

station_crime = (
    crime_df
    .groupby("station_id")["incidents"]
    .sum()
    .reset_index()
)

station_crime = station_crime.merge(
    station_df[
        [
            "station_id",
            "station_name",
            "population",
            "province",
            "urban_type",
            "area_km2",
            "has_taxi_rank",
            "unemployment_rate"
        ]
    ],
    on="station_id",
    how="left"
)

# ----------------------------------------------------------
# Feature engineering for hotspot classification
# ----------------------------------------------------------

station_crime["crime_rate"] = (
    station_crime["incidents"]
    / station_crime["population"]
) * 100000

station_crime["crime_density"] = (
    station_crime["incidents"]
    / station_crime["area_km2"]
)

# 75th percentile hotspot threshold
hotspot_threshold = station_crime["crime_rate"].quantile(0.75)

station_crime["hotspot"] = (
    station_crime["crime_rate"] >= hotspot_threshold
).astype(int)

# ----------------------------------------------------------
# Define predictors and target
# ----------------------------------------------------------

X = station_crime[
    [
        "population",
        "area_km2",
        "has_taxi_rank",
        "unemployment_rate",
        "province",
        "urban_type"
    ]
]

y = station_crime["hotspot"]

# ----------------------------------------------------------
# Train/test split
# ----------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

numeric_features = [
    "population",
    "area_km2",
    "has_taxi_rank",
    "unemployment_rate"
]

categorical_features = [
    "province",
    "urban_type"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# ----------------------------------------------------------
# Create models
# ----------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42
            )
        )
    ]
)

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# ----------------------------------------------------------
# Train models
# ----------------------------------------------------------

logistic_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Predictions

y_pred_logistic = logistic_model.predict(X_test)
y_pred_tree = decision_tree_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)

# ----------------------------------------------------------
# Model evaluation
# ----------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred_logistic),
        accuracy_score(y_test, y_pred_tree),
        accuracy_score(y_test, y_pred_rf)
    ],

    "Precision": [
        precision_score(y_test, y_pred_logistic),
        precision_score(y_test, y_pred_tree),
        precision_score(y_test, y_pred_rf)
    ],

    "Recall": [
        recall_score(y_test, y_pred_logistic),
        recall_score(y_test, y_pred_tree),
        recall_score(y_test, y_pred_rf)
    ],

    "F1 Score": [
        f1_score(y_test, y_pred_logistic),
        f1_score(y_test, y_pred_tree),
        f1_score(y_test, y_pred_rf)
    ]
})

st.subheader("Model Performance Comparison")

st.dataframe(
    model_results,
    use_container_width=True
)

# ----------------------------------------------------------
# Select model
# ----------------------------------------------------------

selected_model = st.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]
)

if selected_model == "Logistic Regression":

    selected_predictions = y_pred_logistic

elif selected_model == "Decision Tree":

    selected_predictions = y_pred_tree

else:

    selected_predictions = y_pred_rf

# ----------------------------------------------------------
# Display selected model metrics
# ----------------------------------------------------------

selected_row = model_results[
    model_results["Model"] == selected_model
].iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{selected_row['Accuracy']:.2%}"
    )

with col2:
    st.metric(
        "Precision",
        f"{selected_row['Precision']:.2%}"
    )

with col3:
    st.metric(
        "Recall",
        f"{selected_row['Recall']:.2%}"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{selected_row['F1 Score']:.2%}"
    )

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

from sklearn.metrics import confusion_matrix

st.subheader(
    f"Confusion Matrix — {selected_model}"
)

cm = confusion_matrix(
    y_test,
    selected_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Non-Hotspot",
        "Actual Hotspot"
    ],
    columns=[
        "Predicted Non-Hotspot",
        "Predicted Hotspot"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)

# ----------------------------------------------------------
# Model interpretation
# ----------------------------------------------------------

st.subheader("Classification Summary")

st.write(
    f"The selected model is **{selected_model}**. "
    f"It achieved an accuracy of "
    f"**{selected_row['Accuracy']:.2%}** on the test dataset."
)

st.write(
    "The confusion matrix shows the number of correctly and "
    "incorrectly classified hotspot and non-hotspot stations. "
    "Recall is particularly important because missing an actual "
    "crime hotspot may affect resource allocation decisions."
)

# ==========================================================
# MODEL VALIDATION
# ==========================================================

st.header("Model Validation")

st.write(
    "Model validation is used to evaluate how consistently the "
    "classification models perform on different subsets of the "
    "training data. Stratified 5-fold cross-validation is used "
    "to maintain the proportion of hotspot and non-hotspot "
    "stations in each fold."
)

from sklearn.model_selection import StratifiedKFold, cross_val_score

# ----------------------------------------------------------
# Create 5-fold stratified cross-validation
# ----------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ----------------------------------------------------------
# Calculate cross-validation scores
# ----------------------------------------------------------

logistic_cv_scores = cross_val_score(
    logistic_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

tree_cv_scores = cross_val_score(
    decision_tree_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

rf_cv_scores = cross_val_score(
    random_forest_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# ----------------------------------------------------------
# Display individual fold scores
# ----------------------------------------------------------

cv_scores_df = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],

    "Logistic Regression": logistic_cv_scores,

    "Decision Tree": tree_cv_scores,

    "Random Forest": rf_cv_scores
})

st.subheader("5-Fold Cross-Validation Scores")

st.dataframe(
    cv_scores_df,
    use_container_width=True
)

# ----------------------------------------------------------
# Calculate mean and standard deviation
# ----------------------------------------------------------

cv_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Mean CV Accuracy": [
        logistic_cv_scores.mean(),
        tree_cv_scores.mean(),
        rf_cv_scores.mean()
    ],

    "Standard Deviation": [
        logistic_cv_scores.std(),
        tree_cv_scores.std(),
        rf_cv_scores.std()
    ]
})

st.subheader("Cross-Validation Performance")

st.dataframe(
    cv_results,
    use_container_width=True
)

# ----------------------------------------------------------
# Select best model
# ----------------------------------------------------------

best_model_name = cv_results.loc[
    cv_results["Mean CV Accuracy"].idxmax(),
    "Model"
]

st.success(
    f"Preferred model based on cross-validation: "
    f"{best_model_name}"
)

# ----------------------------------------------------------
# Interpretation
# ----------------------------------------------------------

st.subheader("Validation Interpretation")

st.write(
    "Cross-validation provides a more reliable estimate of "
    "model performance than relying only on a single train/test "
    "split. The mean accuracy represents the average performance "
    "across the five folds, while the standard deviation shows "
    "how much the performance varies between folds."
)

st.write(
    f"Based on the highest mean cross-validation accuracy, "
    f"**{best_model_name}** is selected as the preferred "
    f"classification model."
)

# ==========================================================
# CRIME FORECASTING
# ==========================================================

st.header("Crime Forecasting")

st.write(
    "A SARIMAX time-series model is used to forecast monthly "
    "crime incidents for the next 12 months. The forecast also "
    "includes a 95% confidence interval to represent uncertainty."
)

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------------------------------------
# Prepare monthly crime time series
# ----------------------------------------------------------

forecast_data = (
    crime_df
    .groupby("month")["incidents"]
    .sum()
    .sort_index()
)

# Make sure the time series has a monthly frequency
forecast_data = forecast_data.asfreq("MS")

# ----------------------------------------------------------
# Fit SARIMAX model
# ----------------------------------------------------------

sarimax_model = SARIMAX(
    forecast_data,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_result = sarimax_model.fit(
    disp=False,
    maxiter=200
)

# ----------------------------------------------------------
# Generate 12-month forecast
# ----------------------------------------------------------

sarimax_forecast = sarimax_result.get_forecast(
    steps=12
)

forecast_mean = sarimax_forecast.predicted_mean

forecast_ci = sarimax_forecast.conf_int()

# ----------------------------------------------------------
# Create forecast table
# ----------------------------------------------------------

forecast_table = pd.DataFrame({
    "Forecast": forecast_mean,
    "Lower CI": forecast_ci.iloc[:, 0],
    "Upper CI": forecast_ci.iloc[:, 1]
})

st.subheader("12-Month Crime Forecast")

st.dataframe(
    forecast_table,
    use_container_width=True
)

# ----------------------------------------------------------
# Forecast graph
# ----------------------------------------------------------

st.subheader("Forecast with 95% Confidence Interval")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    forecast_data.index,
    forecast_data.values,
    label="Historical Crime"
)

ax.plot(
    forecast_mean.index,
    forecast_mean.values,
    label="12-Month Forecast"
)

ax.fill_between(
    forecast_mean.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha=0.2,
    label="95% Confidence Interval"
)

ax.set_xlabel("Date")
ax.set_ylabel("Monthly Crime Incidents")

ax.set_title(
    "12-Month Crime Forecast with 95% Confidence Interval"
)

ax.legend()
ax.grid(True)

fig.tight_layout()

st.pyplot(fig)

# ----------------------------------------------------------
# Forecast explanation
# ----------------------------------------------------------

st.subheader("Forecast Interpretation")

st.write(
    "The SARIMAX model uses historical monthly crime patterns "
    "to estimate crime activity for the next 12 months."
)

st.write(
    "The shaded confidence interval represents the range within "
    "which the future crime values are expected to fall with "
    "approximately 95% confidence, subject to the assumptions "
    "of the model."
)

st.info(
    "Forecast values are estimates and should not be interpreted "
    "as guaranteed future crime levels."
)

# ==========================================================
# CLUSTERING
# ==========================================================

st.header("Crime Station Clustering")

st.write(
    "K-Means clustering is used to group police stations based "
    "on similarities in crime activity and contextual variables. "
    "This is an unsupervised learning technique because no target "
    "variable is provided to the algorithm."
)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# Select clustering features
# ----------------------------------------------------------

cluster_features = station_crime[
    [
        "population",
        "area_km2",
        "unemployment_rate",
        "incidents",
        "crime_rate",
        "crime_density"
    ]
].copy()

# ----------------------------------------------------------
# Standardise features
# ----------------------------------------------------------

cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_features
)

# ----------------------------------------------------------
# Apply K-Means clustering
# ----------------------------------------------------------

kmeans_model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

station_crime["cluster"] = (
    kmeans_model.fit_predict(cluster_scaled)
)

# ----------------------------------------------------------
# Cluster distribution
# ----------------------------------------------------------

st.subheader("Cluster Distribution")

cluster_counts = (
    station_crime["cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)

# ----------------------------------------------------------
# Cluster summary
# ----------------------------------------------------------

st.subheader("Cluster Characteristics")

cluster_summary = (
    station_crime
    .groupby("cluster")[
        [
            "population",
            "area_km2",
            "unemployment_rate",
            "incidents",
            "crime_rate",
            "crime_density"
        ]
    ]
    .mean()
    .round(2)
)

st.dataframe(
    cluster_summary,
    use_container_width=True
)

# ----------------------------------------------------------
# Hotspots per cluster
# ----------------------------------------------------------

st.subheader("Hotspots by Cluster")

cluster_hotspots = (
    station_crime
    .groupby("cluster")["hotspot"]
    .agg(
        Stations="count",
        Hotspots="sum"
    )
)

cluster_hotspots["Hotspot Percentage"] = (
    cluster_hotspots["Hotspots"]
    / cluster_hotspots["Stations"]
) * 100

cluster_hotspots["Hotspot Percentage"] = (
    cluster_hotspots["Hotspot Percentage"]
    .round(2)
)

st.dataframe(
    cluster_hotspots,
    use_container_width=True
)

# ----------------------------------------------------------
# Cluster visualisation
# ----------------------------------------------------------

st.subheader("Crime Rate vs Crime Density")

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    station_crime["crime_rate"],
    station_crime["crime_density"],
    c=station_crime["cluster"],
    s=60
)

ax.set_xlabel(
    "Crime Rate per 100,000 Population"
)

ax.set_ylabel(
    "Crime Density per km²"
)

ax.set_title(
    "Police Station Clusters"
)

ax.grid(True)

st.pyplot(fig)

# ----------------------------------------------------------
# Interpretation
# ----------------------------------------------------------

st.subheader("Clustering Interpretation")

st.write(
    "The clusters represent groups of police stations with "
    "similar combinations of population, geographic area, "
    "unemployment, crime incidents, crime rate and crime density."
)

st.write(
    "Clusters with a high proportion of hotspot stations may "
    "represent groups of stations with stronger crime-related "
    "characteristics. However, clustering identifies patterns "
    "and associations rather than proving that one variable "
    "causes another."
)

# ==========================================================
# DRONE PROGRAMMING SIMULATION
# ==========================================================

st.header("Drone Programming Simulation")

st.write(
    "The drone simulation demonstrates how a drone could visit "
    "identified crime hotspots within a 1 km × 1 km operational "
    "area. The simulation uses generated hotspot coordinates and "
    "a lawnmower-style waypoint pattern."
)

import numpy as np

# ----------------------------------------------------------
# Create hotspot points
# ----------------------------------------------------------

hotspot_stations = station_crime[
    station_crime["hotspot"] == 1
].copy()

# Generate reproducible coordinates inside a 1 km x 1 km grid
np.random.seed(42)

hotspot_stations["x_km"] = np.random.uniform(
    0.05,
    0.95,
    len(hotspot_stations)
)

hotspot_stations["y_km"] = np.random.uniform(
    0.05,
    0.95,
    len(hotspot_stations)
)

# ----------------------------------------------------------
# Display hotspot locations
# ----------------------------------------------------------

st.subheader("Crime Hotspots")

st.dataframe(
    hotspot_stations[
        [
            "station_id",
            "station_name",
            "crime_rate",
            "hotspot",
            "x_km",
            "y_km"
        ]
    ].round(2),
    use_container_width=True
)

# ----------------------------------------------------------
# Create lawnmower waypoint pattern
# ----------------------------------------------------------

waypoints = []

grid_size = 5

for row in range(grid_size):

    y = row * (1 / (grid_size - 1))

    if row % 2 == 0:

        x_values = np.linspace(
            0,
            1,
            grid_size
        )

    else:

        x_values = np.linspace(
            1,
            0,
            grid_size
        )

    for x in x_values:

        waypoints.append(
            (x, y)
        )

# ----------------------------------------------------------
# Plot drone route and hotspots
# ----------------------------------------------------------

st.subheader("Drone Waypoint Route")

fig, ax = plt.subplots(figsize=(9, 7))

waypoint_x = [
    point[0]
    for point in waypoints
]

waypoint_y = [
    point[1]
    for point in waypoints
]

ax.plot(
    waypoint_x,
    waypoint_y,
    marker="o",
    label="Drone Waypoints"
)

ax.scatter(
    hotspot_stations["x_km"],
    hotspot_stations["y_km"],
    s=80,
    label="Crime Hotspots"
)

# Add station labels
for _, row in hotspot_stations.iterrows():

    ax.annotate(
        row["station_id"],
        (
            row["x_km"],
            row["y_km"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xlabel("X Coordinate (km)")
ax.set_ylabel("Y Coordinate (km)")

ax.set_title(
    "1 km × 1 km Drone Crime Hotspot Simulation"
)

ax.legend()
ax.grid(True)

st.pyplot(fig)

# ----------------------------------------------------------
# Drone simulation information
# ----------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Operational Area",
        "1 km × 1 km"
    )

with col2:

    st.metric(
        "Hotspots",
        len(hotspot_stations)
    )

with col3:

    st.metric(
        "Waypoints",
        len(waypoints)
    )

# ----------------------------------------------------------
# Drone method explanation
# ----------------------------------------------------------

st.subheader("Drone Flying Method")

st.write(
    "A lawnmower flight pattern is used because it provides "
    "systematic coverage of the operational area. The drone "
    "travels across one row and then reverses direction on "
    "the next row."
)

st.write(
    "This approach reduces the chance of leaving large areas "
    "uncovered and provides a simple waypoint strategy that "
    "can be implemented programmatically."
)

st.info(
    "The coordinates used in this simulation are generated "
    "for demonstration purposes and do not represent real "
    "geographical police-station locations."
)

# ==========================================================
# TECHNICAL AND NON-TECHNICAL SUMMARY
# ==========================================================

st.header("Project Summary")

# ----------------------------------------------------------
# Non-technical summary
# ----------------------------------------------------------

st.subheader("Non-Technical Summary")

st.write(
    "This dashboard provides an overview of crime patterns "
    "across police stations. Users can explore crime categories, "
    "years, provinces and urban areas using the interactive filters."
)

st.write(
    "Machine learning models are used to identify police stations "
    "that may be classified as crime hotspots. The models use "
    "population, geographic, urban and socioeconomic information "
    "to make the classification."
)

st.write(
    "The forecasting section estimates future monthly crime "
    "activity, while the drone simulation demonstrates how "
    "technology could be used to systematically monitor areas "
    "identified as potential hotspots."
)

# ----------------------------------------------------------
# Technical summary
# ----------------------------------------------------------

st.subheader("Technical Summary")

st.write(
    "The project uses Python, Pandas, NumPy, Scikit-learn, "
    "Statsmodels, Matplotlib and Streamlit."
)

st.write(
    "The classification workflow includes data preprocessing, "
    "feature engineering, categorical encoding, numerical "
    "standardisation, stratified train-test splitting and "
    "machine-learning classification."
)

st.write(
    "Logistic Regression, Decision Tree and Random Forest "
    "models were evaluated using accuracy, precision, recall "
    "and F1 score."
)

st.write(
    "Five-fold Stratified Cross-Validation was used for model "
    "validation. Logistic Regression was selected as the "
    "preferred model based on its cross-validation performance."
)

st.write(
    "K-Means clustering was used to identify groups of stations "
    "with similar crime and contextual characteristics."
)

st.write(
    "SARIMAX was used for time-series forecasting, producing "
    "a 12-month forecast with 95% confidence intervals."
)

st.write(
    "The drone simulation uses a 1 km × 1 km operational grid "
    "and a lawnmower-style waypoint strategy to demonstrate "
    "systematic area coverage."
)

# ----------------------------------------------------------
# Key results
# ----------------------------------------------------------

st.subheader("Key Results")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Hotspots",
        int(station_crime["hotspot"].sum())
    )

with col2:

    st.metric(
        "Stations",
        station_crime["station_id"].nunique()
    )

with col3:

    st.metric(
        "Preferred Model",
        "Logistic Regression"
    )

with col4:

    st.metric(
        "Forecast Period",
        "12 Months"
    )

st.success(
    "The dashboard combines exploratory data analysis, "
    "machine learning, forecasting, clustering and drone "
    "simulation into one integrated crime analytics solution."
)