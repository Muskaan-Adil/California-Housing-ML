# California Housing Dataset – Simple Linear Regression

## Project Overview

This project focuses on building a **Simple Linear Regression model** using the California Housing Prices dataset. The main goal is to predict the `median_house_value` based on `median_income`, and understand the strength of the relationship between income levels and housing prices.

This project demonstrates the essential steps involved in preparing real-world data, cleaning it, selecting features, scaling variables, training a basic machine learning model, and evaluating its performance with metrics and visualizations.

---

## Key Features

- **Data Inspection**: Reviewed the dataset structure, checked column data types, and identified missing values using `info()` and `describe()`.
- **Missing Value Handling**: Filled missing values in `total_bedrooms` using the column median to maintain data consistency.
- **Feature Selection**: Chose `median_income` as the key independent feature for predicting `median_house_value`.
- **Feature Scaling**: Standardized the input feature using `StandardScaler` for better model convergence.
- **Correlation Analysis**: Visualized correlations between numerical variables to validate feature selection.
- **Model Building**: Built a **Simple Linear Regression** model using scikit-learn's `LinearRegression`.
- **Model Evaluation**: Evaluated the model using **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R-squared (R²) score**.
- **Result Visualization**: Created scatter plots showing actual vs. predicted house values and the regression line.

---

## Dataset Information

- **Source**: California Housing Prices Dataset
- **Columns**:
  - `longitude`: Longitude coordinate of the block group.
  - `latitude`: Latitude coordinate of the block group.
  - `housing_median_age`: Median age of houses in the block group.
  - `total_rooms`: Total number of rooms in the block group.
  - `total_bedrooms`: Total number of bedrooms in the block group.
  - `population`: Total population of the block group.
  - `households`: Total number of households in the block group.
  - `median_income`: Median household income of the block group.
  - `median_house_value`: Median house value of the block group (target variable).
  - `ocean_proximity`: Category representing distance to the ocean.

---

## Data Cleaning and Preprocessing Steps

### Loading the Dataset
- Loaded the dataset using pandas and confirmed the first few records with `head()`.

### Inspecting the Data
- Used `info()` and `describe()` to review data types, detect missing values, and get statistical summaries.

### Handling Missing Values
- Detected missing values in the `total_bedrooms` column.
- Replaced missing values with the column's median value using `fillna()`.

### Encoding and Dropping Categorical Variables
- Dropped the `ocean_proximity` categorical column since the initial model focuses on numerical data only.

### Correlation Analysis
- Used a heatmap to study correlations between numerical features.
- Confirmed that `median_income` has a strong positive correlation with `median_house_value`.

### Feature Selection
- Selected `median_income` as the independent feature (X) and `median_house_value` as the dependent variable (y).

### Feature Scaling
- Standardized `median_income` using `StandardScaler` to bring it to a mean 0, variance 1 scale.

### Splitting the Dataset
- Split the data into training and testing sets (80%-20%) using `train_test_split()` with a fixed random state.

### Model Building
- Trained a **Simple Linear Regression** model using the training data.

### Prediction and Evaluation
- Predicted house values on the test set.
- Evaluated performance using:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R-squared Score (R²)**

### Visualization
- Plotted actual vs predicted house values.
- Visualized the fitted regression line against the scatter plot of test data points.

---

## Report

For a more detailed step-by-step explanation of the preprocessing, model building, evaluation, and visualizations, please refer to **[California_Housing_Report.md](California_Housing_Report.md)**.
