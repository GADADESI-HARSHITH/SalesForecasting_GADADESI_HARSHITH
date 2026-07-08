# ==========================================================
# Sales Forecasting & Demand Intelligence Dashboard
# Internship Project - Week 3 & Week 4
# ==========================================================

# Import Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Dashboard Title

st.title("📈 Sales Forecasting & Demand Intelligence Dashboard")

st.markdown("""
This interactive dashboard summarizes the complete Sales Forecasting project.

It includes:

- 📊 Sales Overview
- 📈 Forecast Explorer
- 🚨 Anomaly Detection
- 📦 Product Demand Segmentation

Use the navigation menu on the left to explore different analyses.
""")

# ==========================================================
# Load Dataset
# ==========================================================

sales_df = pd.read_csv("data/train.csv")

sales_df["Order Date"] = pd.to_datetime(
    sales_df["Order Date"],
    dayfirst=True
)

sales_df["Ship Date"] = pd.to_datetime(
    sales_df["Ship Date"],
    dayfirst=True
)

# Monthly Sales

monthly_sales = (
    sales_df
    .groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
    .sum()
    .reset_index()
)

# Weekly Sales

weekly_sales = (
    sales_df
    .groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
    .sum()
    .reset_index()
)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    (
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Detection",
        "Product Segmentation"
    )
)

# ==========================================================
# Sales Overview
# ==========================================================

if page == "Sales Overview":

    st.header("📊 Sales Overview Dashboard")

    # ----------------------------
    # Interactive Filters
    # ----------------------------

    region = st.selectbox(
        "Select Region",
        ["All"] + sorted(sales_df["Region"].unique().tolist())
    )

    category = st.selectbox(
        "Select Category",
        ["All"] + sorted(sales_df["Category"].unique().tolist())
    )

    filtered_df = sales_df.copy()

    if region != "All":
        filtered_df = filtered_df[
            filtered_df["Region"] == region
        ]

    if category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == category
        ]

    # ----------------------------
    # KPI Metrics
    # ----------------------------

    total_sales = filtered_df["Sales"].sum()
    total_orders = filtered_df["Order ID"].nunique()
    total_customers = filtered_df["Customer ID"].nunique()
    total_products = filtered_df["Product Name"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
    col2.metric("🛒 Orders", total_orders)
    col3.metric("👥 Customers", total_customers)
    col4.metric("📦 Products", total_products)

    st.divider()

    # ----------------------------
    # Total Sales by Year
    # ----------------------------

    st.subheader("📊 Total Sales by Year")

    year_sales = (
        filtered_df
        .groupby(filtered_df["Order Date"].dt.year)["Sales"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(10,4))

    ax.bar(
        year_sales.index.astype(str),
        year_sales.values
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    st.divider()

    # ----------------------------
    # Monthly Sales Trend
    # ----------------------------

    st.subheader("📈 Monthly Sales Trend")

    monthly_filtered = (
        filtered_df
        .groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        monthly_filtered["Order Date"],
        monthly_filtered["Sales"],
        marker="o",
        linewidth=2
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    st.divider()

    # ----------------------------
    # Category Sales
    # ----------------------------

    st.subheader("📦 Sales by Category")

    category_sales = (
        filtered_df
        .groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        category_sales.index,
        category_sales.values
    )

    ax.set_ylabel("Sales")

    st.pyplot(fig)

    st.divider()

    # ----------------------------
    # Region Sales
    # ----------------------------

    st.subheader("🌍 Sales by Region")

    region_sales = (
        filtered_df
        .groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        region_sales.index,
        region_sales.values
    )

    ax.set_ylabel("Sales")

    st.pyplot(fig)

    st.divider()

    st.subheader("📄 Dataset Preview")

    st.dataframe(filtered_df.head())

    
# ==========================================================
# Forecast Explorer
# ==========================================================

elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    st.write(
        "This section compares the forecasts generated using "
        "SARIMA, Facebook Prophet, and XGBoost models."
    )

    model = st.selectbox(
        "Select Forecasting Model",
        [
            "SARIMA",
            "Facebook Prophet",
            "XGBoost"
        ]
    )

    forecast_horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    if model == "SARIMA":

        st.subheader("SARIMA Forecast")

        st.image(
            "charts/sarima_forecast.png",
            use_container_width=True
        )

        st.metric("MAE", "20,581")
        st.metric("RMSE", "22,191")
        st.metric("MAPE", "0.2194")

        st.success(
            "SARIMA effectively captures trend and seasonality using statistical time-series modelling."
        )

    elif model == "Facebook Prophet":

        st.subheader("Facebook Prophet Forecast")

        st.image(
            "charts/prophet_forecast.png",
            use_container_width=True
        )

        st.metric("MAE", "20,251")
        st.metric("RMSE", "22,318")
        st.metric("MAPE", "0.2186")

        st.success(
            "Facebook Prophet automatically models trend and yearly seasonality."
        )

    else:

        st.subheader("XGBoost Forecast")

        st.image(
            "charts/xgboost_forecast.png",
            use_container_width=True
        )

        st.metric("MAE", "14,458")
        st.metric("RMSE", "18,905")
        st.metric("MAPE", "0.1401")

        st.success(
            "XGBoost achieved the lowest forecasting error among all three models."
        )

    st.write(f"Showing a {forecast_horizon}-month forecast.")
# ==========================================================
# Anomaly Detection
# ==========================================================

elif page == "Anomaly Detection":

    st.header("🚨 Sales Anomaly Detection")

    st.write(
        "This page highlights unusual sales behaviour detected "
        "using statistical anomaly detection (Z-Score)."
    )

    st.image(
        "charts/sales_anomalies.png",
        use_container_width=True
    )

    anomaly_df = pd.DataFrame({
        "Order Date": ["2018-11-30"],
        "Sales": [117938.155],
        "Z-Score": [2.8653],
        "Status": ["Anomaly"]
    })

    st.subheader("Detected Anomalies")

    st.dataframe(anomaly_df)

    st.metric(
        "Number of Anomalies",
        len(anomaly_df)
    )

    st.success(
        "One significant positive anomaly was detected. "
        "This spike may correspond to seasonal demand or a successful promotional campaign."
    )

# ==========================================================
# Product Segmentation
# ==========================================================

elif page == "Product Segmentation":

    st.header("📦 Product Demand Segmentation")

    st.write(
        "Products are grouped into demand clusters using K-Means clustering."
    )

    st.image(
        "charts/product_segmentation.png",
        use_container_width=True
    )

    cluster_summary = pd.DataFrame({

        "Cluster":[
            "Low Demand",
            "High Demand",
            "Medium Demand"
        ],

        "Products":[
            1655,
            16,
            178
        ],

        "Average Sales":[
            550.33,
            21114.80,
            5690.47
        ]

    })

    st.subheader("Cluster Summary")

    st.dataframe(cluster_summary)

    st.success(
        """
High-demand products generate the largest revenue and should receive higher inventory allocation.

Medium-demand products require balanced inventory planning.

Low-demand products may benefit from promotional strategies.
"""
    )
