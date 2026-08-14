# House Price Prediction with Linear Regression

## Project Overview

This project develops an end-to-end machine learning solution to predict house prices using Linear Regression and regularized regression models.

The project covers the complete machine learning workflow, including data loading, exploratory data analysis, data cleaning, feature selection, categorical encoding, model training, evaluation, visualization, model interpretation, and deployment through an interactive Streamlit dashboard.

The Ames Housing dataset is used for this project, with `SalePrice` as the target variable.

---

## Objective

The main objective is to build and evaluate a machine learning model that predicts house prices using features related to:

* House size
* Overall quality
* Construction year
* Basement area
* Garage capacity
* Number of bathrooms
* Number of bedrooms
* Neighborhood/location

The project also compares Linear Regression with Ridge and Lasso Regression to determine whether regularization improves prediction performance.

---

## Technologies Used

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* Streamlit

---

## Dataset

The Ames Housing dataset contains 1,460 house records and 81 original features, along with the target variable `SalePrice`.

Selected features used for modelling include:

* `GrLivArea`
* `OverallQual`
* `YearBuilt`
* `TotalBsmtSF`
* `GarageCars`
* `FullBath`
* `BedroomAbvGr`
* `Neighborhood`

---

## Project Workflow

### 1. Exploratory Data Analysis

The dataset was examined using:

* Dataset shape and structure
* Data types
* Missing-value analysis
* Descriptive statistics
* House price distribution
* Feature distributions
* Correlation analysis

### 2. Feature Selection

Features were selected based on their expected relationship with house prices.

For example:

* `GrLivArea` represents above-ground living area and is expected to have a strong relationship with price.
* `OverallQual` represents the overall quality of the property and is an important indicator of house value.
* `YearBuilt` captures the age of the property.
* `TotalBsmtSF` represents basement area.
* `GarageCars` represents garage capacity.
* `FullBath` and `BedroomAbvGr` represent important house characteristics.
* `Neighborhood` captures location-related differences in house prices.

### 3. Data Preprocessing

The following preprocessing steps were performed:

* Removed the unnecessary index column
* Selected relevant numerical and categorical features
* Filled missing numerical values using median imputation
* Filled missing categorical values using the most frequent category
* Applied One-Hot Encoding to the `Neighborhood` feature

A scikit-learn preprocessing pipeline was used to keep data transformation and model training consistent.

### 4. Train/Test Split

The dataset was divided into:

* 80% training data
* 20% testing data

A fixed random state of 42 was used for reproducibility.

### 5. Linear Regression

A Linear Regression model was trained using a scikit-learn preprocessing and modelling pipeline.

### 6. Model Evaluation

The Linear Regression model was evaluated using:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Linear Regression Results

| Metric | Value |
|---|---:|
| MSE | 1,301,513,687.00 |
| RMSE | 36,076.50 |
| R² Score | 0.8303 |

The R² score of approximately 0.83 indicates that the Linear Regression model explains around 83% of the variation in house prices in the test dataset.

---

## Visualizations

The project includes:

* House price distribution
* Correlation heatmap
* Actual vs predicted price scatter plot
* Residual plot
* Regression coefficient analysis

These visualizations help evaluate model performance, identify relationships between features, and understand prediction errors.

---

## Coefficient Analysis

Linear Regression coefficients were analyzed to identify features with relatively stronger positive and negative effects on predicted house prices.

For categorical `Neighborhood` features, coefficients represent differences relative to the encoded baseline category.

Coefficient analysis provides an interpretable view of how individual features influence the model's predictions.

---

## Bonus: Regularized Regression

The Linear Regression model was compared with Ridge and Lasso Regression.

| Model | MSE | RMSE | R² |
|---|---:|---:|---:|
| Ridge Regression | 1,300,277,978.51 | 36,059.37 | 0.83 |
| Linear Regression | 1,301,513,687.00 | 36,076.50 | 0.83 |
| Lasso Regression | 1,301,882,351.90 | 36,081.61 | 0.83 |

### Best Model

Ridge Regression achieved the lowest MSE and RMSE among the three models on the test dataset.

However, the difference between the models is small, indicating that regularization provides only a modest improvement for the selected feature set.

---

## Interactive Streamlit Dashboard

The project also includes an interactive Streamlit dashboard that provides a user-friendly interface for house price prediction.

The dashboard allows users to enter property details and receive an estimated house price using the trained Ridge Regression model.

### Property Inputs

Users can provide:

* Living Area
* Overall Quality
* Year Built
* Basement Area
* Garage Capacity
* Number of Bathrooms
* Number of Bedrooms
* Neighborhood

### Dashboard Features

The interactive application provides:

* Interactive house price prediction
* Estimated house price
* Linear Regression performance
* Ridge Regression performance
* Lasso Regression performance
* MSE comparison
* RMSE comparison
* R² score comparison
* Actual vs Predicted price visualization
* Residual analysis
* Dataset overview

---

## Running the Streamlit Dashboard

### 1. Create a Virtual Environment

After downloading or cloning the project, open a terminal inside the project folder and run:

```bash
python3 -m venv .venv