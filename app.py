import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    margin-bottom: 2rem;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    color: #cbd5e1;
}

.metric-card {
    padding: 1rem;
    border-radius: 14px;
    background: white;
    border: 1px solid #e2e8f0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/house_prices.csv")


df = load_data()


# ============================================================
# FEATURES
# ============================================================

selected_features = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "BedroomAbvGr",
    "Neighborhood"
]

target = "SalePrice"

model_df = df[selected_features + [target]].copy()

X = model_df[selected_features]
y = model_df[target]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# PREPROCESSING
# ============================================================

numerical_features = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "BedroomAbvGr"
]

categorical_features = [
    "Neighborhood"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# MODELS
# ============================================================

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

ridge_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0))
    ]
)

lasso_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Lasso(alpha=100.0, max_iter=10000))
    ]
)


# Train models
linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)


# Predictions
linear_pred = linear_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)


# Metrics
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_pred)

ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, ridge_pred)

lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_r2 = r2_score(y_test, lasso_pred)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🏠 House Price Predictor</h1>
    <p>
        Machine Learning powered house price prediction using
        Linear, Ridge and Lasso Regression.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏠 Property Details")

st.sidebar.markdown(
    "Enter the house characteristics below to estimate its sale price."
)

gr_liv_area = st.sidebar.number_input(
    "Living Area (sq ft)",
    min_value=300,
    max_value=6000,
    value=1500
)

overall_qual = st.sidebar.slider(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=6
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000
)

total_bsmt_sf = st.sidebar.number_input(
    "Basement Area (sq ft)",
    min_value=0,
    max_value=4000,
    value=800
)

garage_cars = st.sidebar.slider(
    "Garage Capacity",
    min_value=0,
    max_value=5,
    value=2
)

full_bath = st.sidebar.slider(
    "Full Bathrooms",
    min_value=0,
    max_value=5,
    value=2
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    min_value=0,
    max_value=8,
    value=3
)

neighborhood = st.sidebar.selectbox(
    "Neighborhood",
    sorted(df["Neighborhood"].dropna().unique())
)


# ============================================================
# PREDICTION
# ============================================================

input_data = pd.DataFrame({
    "GrLivArea": [gr_liv_area],
    "OverallQual": [overall_qual],
    "YearBuilt": [year_built],
    "TotalBsmtSF": [total_bsmt_sf],
    "GarageCars": [garage_cars],
    "FullBath": [full_bath],
    "BedroomAbvGr": [bedrooms],
    "Neighborhood": [neighborhood]
})


st.subheader("💰 House Price Prediction")

if st.button("🔮 Predict House Price", use_container_width=True):

    prediction = ridge_model.predict(input_data)[0]

    st.success("Prediction generated successfully!")

    st.metric(
        "Estimated House Price",
        f"${prediction:,.0f}"
    )

    st.caption(
        "Prediction generated using the Ridge Regression model."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Linear Regression R²",
        f"{linear_r2:.4f}"
    )

with col2:
    st.metric(
        "Ridge Regression R²",
        f"{ridge_r2:.4f}"
    )

with col3:
    st.metric(
        "Lasso Regression R²",
        f"{lasso_r2:.4f}"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("🏆 Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],
    "MSE": [
        linear_mse,
        ridge_mse,
        lasso_mse
    ],
    "RMSE": [
        linear_rmse,
        ridge_rmse,
        lasso_rmse
    ],
    "R²": [
        linear_r2,
        ridge_r2,
        lasso_r2
    ]
})

st.dataframe(
    comparison.style.format({
        "MSE": "{:,.2f}",
        "RMSE": "{:,.2f}",
        "R²": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.subheader("📈 Actual vs Predicted Prices")

fig1, ax1 = plt.subplots(figsize=(9, 6))

ax1.scatter(
    y_test,
    ridge_pred,
    alpha=0.7
)

min_price = min(y_test.min(), ridge_pred.min())
max_price = max(y_test.max(), ridge_pred.max())

ax1.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

ax1.set_xlabel("Actual Price")
ax1.set_ylabel("Predicted Price")
ax1.set_title("Ridge Regression: Actual vs Predicted")

st.pyplot(fig1)


# ============================================================
# RESIDUAL PLOT
# ============================================================

st.subheader("📉 Residual Analysis")

residuals = y_test - ridge_pred

fig2, ax2 = plt.subplots(figsize=(9, 5))

ax2.scatter(
    ridge_pred,
    residuals,
    alpha=0.7
)

ax2.axhline(
    y=0,
    linestyle="--"
)

ax2.set_xlabel("Predicted Price")
ax2.set_ylabel("Residual")
ax2.set_title("Ridge Regression Residual Plot")

st.pyplot(fig2)


# ============================================================
# DATASET SUMMARY
# ============================================================

st.divider()

st.subheader("📋 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Houses", f"{len(df):,}")
c2.metric("Features", f"{df.shape[1]}")
c3.metric("Training Samples", f"{len(X_train):,}")
c4.metric("Testing Samples", f"{len(X_test):,}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "House Price Prediction • Machine Learning Project • "
    "Python + Pandas + Scikit-learn + Streamlit"
)