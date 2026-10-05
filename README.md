\# Crime Analytics in South Africa



\## Project Overview



This project presents a machine learning and data analytics solution for exploring crime patterns in South Africa.



The system combines crime incident data with police-station demographic, geographic and socioeconomic information. Exploratory Data Analysis, feature engineering, machine learning classification, model validation, clustering, time-series forecasting and drone simulation are used to demonstrate a complete data-driven crime analytics workflow.



\## Objectives



The main objectives of the project are to:



\* Explore crime patterns across police stations and time.

\* Identify potential crime hotspots.

\* Apply machine learning classification models.

\* Evaluate and validate classification models.

\* Group police stations using clustering.

\* Forecast future monthly crime activity.

\* Provide an interactive Streamlit dashboard.

\* Demonstrate a drone-based hotspot monitoring simulation.

\* Apply appropriate coding and analytical practices.



\## Technologies Used



\* Python

\* Jupyter Notebook

\* Pandas

\* NumPy

\* Matplotlib

\* Scikit-learn

\* Statsmodels

\* Streamlit



\## Machine Learning Models



The project evaluates three classification algorithms:



1\. Logistic Regression

2\. Decision Tree

3\. Random Forest



The models are evaluated using:



\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* Confusion Matrix

\* Stratified 5-Fold Cross-Validation



Logistic Regression was selected as the preferred model based on its cross-validation performance.



\## Clustering



K-Means clustering is used to group police stations according to similarities in crime activity and contextual characteristics.



Four clusters are used in the final analysis.



\## Forecasting



Monthly crime activity is analysed using time-series forecasting.



A SARIMAX model is used to produce a 12-month forecast with 95% confidence intervals.



\## Streamlit Dashboard



The Streamlit dashboard provides:



\* Interactive province, crime category and year filters.

\* Crime EDA visualisations.

\* Machine learning model comparison.

\* Confusion matrix.

\* Model validation results.

\* Crime forecasting.

\* K-Means clustering results.

\* Drone programming simulation.

\* Technical and non-technical project summaries.



\## Drone Simulation



A simulated 1 km × 1 km operational area is used to demonstrate drone monitoring of crime hotspots.



A lawnmower-style waypoint pattern is used to provide systematic coverage of the area.



The coordinates in the simulation are generated for demonstration purposes and do not represent real geographical police-station coordinates.



\## Project Files



\* `crime\_analysis.ipynb` — Main Jupyter Notebook containing the complete analysis.

\* `app.py` — Streamlit dashboard application.

\* `station\_clean.csv` — Cleaned police station/context dataset.

\* `crime\_clean.csv` — Cleaned crime dataset.

\* `requirements.txt` — Python dependencies.



\## How to Run the Project



\### 1. Install the required libraries



```bash

pip install -r requirements.txt

```



\### 2. Open the Jupyter Notebook



```bash

jupyter notebook

```



Open:



```text

crime\_analysis.ipynb

```



\### 3. Run the Streamlit Dashboard



From the project directory:



```bash

streamlit run app.py

```



The dashboard will open in a web browser.



\## Important Note



The datasets used during development are fictional practice datasets created for demonstrating the machine learning workflow. For the final examination submission, factual datasets and their direct source links should be documented according to the examination requirements.



\## Conclusion



The project demonstrates how machine learning and data analytics techniques can be integrated into a crime analytics system. The combination of exploratory analysis, hotspot classification, model validation, clustering, forecasting, an interactive dashboard and drone simulation provides a complete prototype for data-driven crime analysis.



