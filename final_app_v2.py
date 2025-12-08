# final_streamlit_app.py
# CMSE 830 Final Project Streamlit App
# Updated structure based on mid-semester app
#this app was generated with the help of Chat-gpt version 5, accessed 11/10/25

import streamlit as st
st.set_page_config(page_title="Health Data Project", layout="wide")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------
# Load Data
# ---------------------------

life_exp_df   = pd.read_csv("data/Life-Expectancy-Data-Updated.csv")
gov_metrics_df = pd.read_csv("data/World_Bank_data.csv")
healthcare_access_df = pd.read_csv("data/HealthcareAccess.csv")
water_quality_df = pd.read_csv("data/WaterQuality.csv")
expenditure_df   = pd.read_csv("data/Add_Gov_Expenditure.csv")

imputed_df          = pd.read_csv("data/imputed_df.csv")
merged_df           = pd.read_csv("data/merged_df.csv")
imputed_encoded_df  = pd.read_csv("data/imputed_encoded_df.csv")


# Local paths (you can swap to URL version if you prefer)
#life_exp_df   = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/Final Project/Life-Expectancy-Data-Updated.csv")
#gov_metrics_df = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/Final Project/World_Bank_data.csv")
#healthcare_access_df = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/Final Project/HealthcareAccess.csv")
#water_quality_df = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/Final Project/WaterQuality.csv")
#expenditure_df= pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/Final Project/Add_Gov_Expenditure.csv")

#imputed_df    = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/Final Project/imputed_df.csv")
#merged_df     = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/Final Project/merged_df.csv")
#imputed_encoded_df = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/Final Project/imputed_encoded_df.csv")



@st.cache_resource
def get_model_artifacts(imputed_df: pd.DataFrame):
    """
    Train and cache the Linear Regression and Random Forest models,
    along with performance metrics and a test-set results DataFrame.

    This runs only once (unless the data change) thanks to caching.
    """
    df = imputed_df.copy()

    # Meta columns we want to keep for interpretation
    meta_cols = [c for c in ["country", "region", "year"] if c in df.columns]

    if "life_expectancy" not in df.columns:
        raise ValueError("life_expectancy column not found in imputed_df.")

    # Build X and y for modeling
    drop_cols = meta_cols + ["life_expectancy"]
    X = df.drop(columns=drop_cols)
    y = df["life_expectancy"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Linear Regression (with scaling) ----
    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_test_lr = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_lr, y_train)
    y_pred_lr = lr.predict(X_test_lr)

    r2_lr = r2_score(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    lr_coef = lr.coef_

    # ---- Random Forest (no scaling) ----
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    r2_rf = r2_score(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    rf_importances = rf.feature_importances_

    # ---- Test-set results for interactive modeling ----
    results_df = pd.DataFrame(
        {
            "life_expectancy_actual": y_test,
            "life_expectancy_pred_lr": y_pred_lr,
            "life_expectancy_pred_rf": y_pred_rf,
        },
        index=y_test.index,
    )

    if meta_cols:
        results_df[meta_cols] = df.loc[results_df.index, meta_cols]

    return {
        "X_columns": X.columns.tolist(),
        "lr": lr,
        "rf": rf,
        "scaler": scaler,
        "r2_lr": r2_lr,
        "rmse_lr": rmse_lr,
        "r2_rf": r2_rf,
        "rmse_rf": rmse_rf,
        "lr_coef": lr_coef,
        "rf_importances": rf_importances,
        "results_df": results_df,
    }

# For the final app, treat df_imputed as the "post-MICE, clipped" dataset
imputed_df = imputed_df.copy()

# ---------------------------
# Page Setup
# ---------------------------

st.sidebar.title("Navigation")
pages = [
    "Project Overview",
    "Data Processing",
    "EDA and Modeling",
    "Interactive Page",
    "Importance & Conclusion"
]
selection = st.sidebar.radio("Select Page:", pages)

st.sidebar.markdown("---")
st.sidebar.subheader("📘 Page Guide")

st.sidebar.markdown(
    """
    **Project Overview** – Goals, motivation, and project roadmap  

    **Data Processing** – Cleaning, missingness, and imputation

    **EDA & Modeling Results** – EDA and modeling process

    **Interactive Exploration** – Interactive plots and scenario simulator  
    
    **Importance & Conclusion** – Key takeaways and policy implications  
    """
)


# =========================================================
# Page 1: Project Overview
# =========================================================
def project_overview():
    st.title("🌍 Health & Government Expenditure")
    st.caption("CMSE 830 Final Project — By: Luke Hudak")

    st.markdown("---")

    # ===============================
    # 1. FULL-WIDTH PROJECT GOAL
    # ===============================
    st.subheader("🎯 Project Goal")
    st.markdown(
        """
       The goals of this project is to investigates how **government expenditure**, **health system capacity**, and **basic public resources** (such as access to safe water) relate to **life expectancy worldwide**.
        
        The objective is to understand:
        - Which factors most strongly align with a longer lifespan
        - How health and economic factors differ from region to region 
        - Be able to apply ML model to predict lifespan  
        """
    )

    st.markdown("---")

    # ===============================
    # 2. PROJECT ROADMAP
    # ===============================
    st.subheader("🗺️ Project Roadmap")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown(
            """
            **1️⃣ Data Preparation**  
            - Merge datasets  
            - Missingness  
            - Imputation  
            - Encoding & clipping  
            """
        )

    with step2:
        st.markdown(
            """
            **2️⃣ EDA and Modeling**  
            - Summary statistics  
            - Distributions & correlations  
            - PCA patterns  
            - Linear regression and Random Forest
            """
        )

    with step3:
        st.markdown(
            """
            **3️⃣ Interpretation**  
            - Identify key drivers  
            - Assess spending & capacity effects  
            - Summarize insights & caveats  
            """
        )

    with step4:
        st.markdown(
            """
            **4️⃣ Interactive Element**  
            - Interactive visualizations
            - Explore the data yourself 
            - Scenario simulator
            """
        )

    st.markdown("---")

    # ===============================
    # 3. FULL-WIDTH IMPORTANCE SECTION
    # ===============================
    st.subheader("📌 Why This Analysis Matters")
    st.markdown(
        """
        **Life expectancy** is shaped by economic conditions, healthcare systems, environmental factors, and access to basic services, among other things.

        Understanding these relationships  can help:
        - Policymakers make better spending decisions
        - Identify potential areas of investment (e.g., water access, health workforce)  
        - Reveal global inequalities in health outcomes   
        """
    )

    st.markdown("---")


# =========================================================
# Page 2: Data Processing
# =========================================================
def data_processing():
    st.title("⚙️ Data Processing")

    # ===============================
    # Brief roadmap-style summary
    # ===============================
    st.subheader("Summary:")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            **1️⃣ Collect & Merge**  
            - Load multiple global datasets  
            - Align by `country` and `year`  
            - Build a unified dataframe 
            """
        )

    with c2:
        st.markdown(
            """
            **2️⃣ Investigate Missingness**  
            - Identify incomplete variables  
            - Visualize missing patterns  
            - Decide what to keep or drop  
            """
        )

    with c3:
        st.markdown(
            """
            **3️⃣ Impute & Clean**  
            - Apply MICE for numeric gaps  
            - Clip to realistic ranges    
            """
        )

    with c4:
        st.markdown(
            """
            **4️⃣ Encode & Scale**  
            - One-hot encode `region`  
            - Standardize predictors for PCA / modeling  
            - Preserve robustness for tree models  
            """
        )

    st.markdown("---")

    # ===============================
    # Tabs for each detailed section
    # ===============================
    tab_data_sources, tab_raw_merge, tab_missing, tab_impute, tab_encode = st.tabs(
        ["Data & Sources", "Raw Datasets & Merging", "Missingness", "Imputation & Clipping", "Encoding & Model-Ready"]
    )

    # -------------------------------------------
    # Tab 1: Data & Sources (list with 5 slots)
    # -------------------------------------------
    with tab_data_sources:
        st.subheader("📂 Data & Sources")

        st.markdown(
            """
            Below are the core datasets used in this project, along with their sources and brief descriptions.

            **1. Life Expectancy Dataset (WHO / Kaggle)**  
            Source: https://www.kaggle.com/datasets/lashagoch/life-expectancy-who-updated  
            - Contains life expectancy values and key health indicators across countries and years  
            - Includes mortality variables, vaccination coverage, BMI, alcohol use, HIV incidence, etc.  

            **2. Government Metrics Dataset (World Bank)**  
            Source: https://data.worldbank.org/indicator  
            - Includes economic and public spending indicators  
            - Examples: GDP per capita, government health expenditure, R&D expenditure, population, consumption expenditure  

            **3. Healthcare Access Data (World Health Organization)**  
            Source: https://data.who.int  
            - Includes access to essential health services and workforce/infrastructure indicators  
            - Examples: physicians per 1000, nurses, hospital beds, service coverage  

            **4. Water Quality & Access Dataset (WHO/UNICEF Joint Monitoring Programme – JMP)**  
            Source: https://washdata.org  
            - Tracks access to safe drinking water, sanitation, and hygiene  
            - Provides national estimates of safely managed water sources and related public health indicators  

            **5. Additional Government Expenditure Dataset (World Bank)**  
            Source: https://data.worldbank.org/indicator  
            - Supplementary government spending variables not included in Dataset 2  
            - Includes targeted health expenditure components for more granular modeling  
            """
            )


    # -------------------------------------------
    # Tab 2: Raw Datasets & Merging
    # -------------------------------------------
    with tab_raw_merge:
        st.subheader("🔹 Raw Datasets & Merging")

        st.markdown(
            """
            **Merging logic:**  
            - Datasets are merged on **`country`** and **`year`**, using the life expectancy dataset as the base.  
            - This preserves all available rows with `life_expectancy` and attaches matching government, health system, and resource indicators.  
            - Additional datasets (e.g., health access, water quality, expenditure) are merged using the same keys wherever possible.  
            - The result is a **single dataframe** that can be used for EDA, PCA, and modeling.  
            """
        )

        st.markdown("#### Life Expectancy Dataset (raw sample)")
        st.dataframe(life_exp_df.head())

        st.markdown("#### Merged Dataset (pre-imputation)")
        st.dataframe(merged_df.head())

    # -------------------------------------------
    # Tab 3: Missingness (heatmap only)
    # -------------------------------------------
    with tab_missing:
        st.subheader("🔹 Missingness Overview")

        st.markdown(
            """
            Before modeling, it is important to understand **where data are missing** and how severe the missingness is across variables.
            """
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(merged_df.isna().T, cmap="magma", cbar=False)
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Columns")
        ax.set_title("Missingness Heatmap (Merged Data)")
        st.pyplot(fig)

        st.markdown(
            """
            **Interpretation:**  
            - Some variables (e.g., certain workforce or environment indicators) exhibit very high missingness and are excluded from final modeling.  
            - Variables with moderate missingness are retained but **imputed** using a multivariate method (MICE), rather than dropped entirely.  
            - This helps maintain sample size while still controlling for data quality.  
            """
        )

    # -------------------------------------------
    # Tab 4: Imputation & Clipping
    # -------------------------------------------
    with tab_impute:
        st.subheader("🔹 Imputation with MICE & Clipping")

        st.markdown(
            """
            **Imputation strategy:**  
            - Use **MICE (Multivariate Imputation by Chained Equations)** from sklearn to fill in missing numeric values.  
            - Each variable with missing data is modeled as a function of the others, and the process is iterated multiple times.  
            - Numeric features are temporarily **standardized**, imputed, then **inverse-transformed** back to their original units.  
            - After imputation, values are **clipped to realistic ranges**, for example:
                - Percentages constrained to [0, 100]  
                - Counts and rates constrained to be ≥ 0  
            """
        )

        st.markdown("#### Sample of post-imputation dataset (used for EDA and modeling)")
        st.dataframe(imputed_df.head())

    # -------------------------------------------
    # Tab 5: Encoding & Model-Ready Data
    # -------------------------------------------
    with tab_encode:
        st.subheader("🔹 Encoding & Model-Ready Data")

        st.markdown(
            """
            **Region encoding:**  
            - The categorical variable `region` is converted to multiple binary (0/1) columns via one-hot encoding.    

            **Scaling:**  
            - For **PCA and Linear Regression**, numeric predictors are standardized using `StandardScaler`.  
            - For **Random Forest**, scaling is **not required**, since tree-based models are scale-invariant.  

            Together, these steps produce a **clean, consistent feature matrix** that can be used across  
            different modeling approaches without redoing the data preparation.
            """
        )

        st.markdown(
            """
            The **`imputed_df`** dataset shown in the previous tab is the main source for  
            **exploratory analysis**, **PCA**, and **predictive modeling** in the rest of the app.
            """
        )


# =========================================================
# Page 3: Exploratory Data Analysis (EDA + PCA + Interactive)
# =========================================================
def eda_model_results_page():
    st.title("📈 EDA & Modeling Results")

    st.markdown(
        """
        This page presents my EDA and modeling done for this project:

        - **Descriptive EDA:** summary statistics, distributions, correlations, and PCA  
        - **Modeling Results:** performance and key factors from Linear Regression and Random Forest  
        """
    )

    df = imputed_df.copy()

    tab_static, tab_model = st.tabs(
        ["📘 Descriptive EDA", "🤖 Modeling Results"]
    )

    # =====================================================
    # TAB 1: DESCRIPTIVE EDA
    # =====================================================
    with tab_static:
        st.subheader("Descriptive EDA")

        st.markdown(
            """
            This section summarizes the main exploratory analyses used in the project:

            - **Summary statistics** to understand typical values and spread  
            - **Distributions and boxplots** to see distribution, outliers, and regional differences  
            - **Correlation** to see how key variables are related 
            - **PCA** to reduce dimensionality and identify dominant patterns across many variables  
            """
        )

        sub1, sub2, sub3, sub4 = st.tabs(
            ["Summary Statistics", "Distributions & Boxplots", "Correlation", "PCA"]
        )

        # ----- Summary Statistics -----
        with sub1:
            st.subheader("Summary Statistics")

            key_vars = [
                "life_expectancy",
                "infant_deaths",
                "under_five_deaths",
                "adult_mortality",
                "gdp_per_capita",
                "education_expenditure_pct_gdp",
                "rnd_expenditure_pct_gdp",
                "gen_gov_health_expenditure_pct_gdp",
                "gen_gov_health_expenditure_pct_gov_exp",
                "access_to_safe_water_pct_pop",
            ]
            existing = [c for c in key_vars if c in df.columns]
            if existing:
                st.dataframe(df[existing].describe().T)
            else:
                st.info("No key variables found for summary statistics.")

        # ----- Distributions & Boxplots -----
        with sub2:
            st.subheader("Distributions & Boxplots")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            region_dummies = [
                "Africa",
                "Asia",
                "Central America and Caribbean",
                "European Union",
                "Middle East",
                "North America",
                "Oceania",
                "Rest of Europe",
                "South America",
            ]
            numeric_main = [c for c in numeric_cols if c not in region_dummies]

            show_cols = numeric_main[:9]
            if show_cols:
                fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                axes = axes.flatten()
                for i, col in enumerate(show_cols):
                    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
                    axes[i].set_title(col.replace("_", " ").title())
                for j in range(len(show_cols), len(axes)):
                    axes[j].axis("off")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No numeric variables available for histogram display.")

            if "region" in df.columns:
                fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
                sns.boxplot(x="region", y="life_expectancy", data=df, ax=axes2[0])
                axes2[0].set_title("Life Expectancy by Region")

                if "gen_gov_health_expenditure_pct_gdp" in df.columns:
                    sns.boxplot(
                        x="region",
                        y="gen_gov_health_expenditure_pct_gdp",
                        data=df,
                        ax=axes2[1],
                    )
                    axes2[1].set_title("Gov Health Expenditure (% GDP) by Region")
                else:
                    axes2[1].axis("off")

                for ax in axes2:
                    ax.tick_params(axis="x", rotation=45)
                plt.tight_layout()
                st.pyplot(fig2)

        # ----- Correlation -----
        with sub3:
            st.subheader("Correlation Heatmap")

            corr_vars = [
                "life_expectancy",
                "infant_deaths",
                "under_five_deaths",
                "adult_mortality",
                "gdp_per_capita",
                "education_expenditure_pct_gdp",
                "rnd_expenditure_pct_gdp",
                "gen_gov_health_expenditure_pct_gdp",
                "access_to_safe_water_pct_pop",
            ]
            corr_vars = [c for c in corr_vars if c in df.columns]

            if len(corr_vars) >= 2:
                corr = df[corr_vars].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    square=True,
                    cbar=True,
                    linewidths=0.5,
                    annot_kws={"size": 9},
                )
                ax.set_title("Correlation Heatmap (Key Variables)", fontsize=14)
                st.pyplot(fig)
            else:
                st.info("Not enough variables for correlation heatmap.")

        # ----- PCA (images) -----
                # ----- PCA (computed in app) -----
                # ----- PCA (static images from notebook) -----
        with sub4:
            st.subheader("Principal Component Analysis (PCA)")

            st.markdown(
                """
                PCA is used to **summarize variation across many correlated predictors** into a smaller
                number of components, and to visualize how variables and observations relate in a reduced
                dimensional space.

                The two plots below are taken directly from the analysis notebook:
                - A **scree plot**, showing how much variance each principal component explains  
                - A **biplot**, showing how countries (points) and variables (arrows) align along PC1 and PC2  
                """
            )

            st.image(
                "data/scree.png",
                caption="PCA Scree Plot: Proportion of variance explained",
                use_container_width=True )

            # Biplot (bottom)
            st.image(
                "data/biplot.png",
                caption="PCA Biplot: PC1 vs PC2 (countries and variable loadings)",
                use_container_width=True,
            )

            st.caption(
                "These figures are exported from the analysis notebook to ensure consistency "
                "between the app and the report."
            )



    # =====================================================
    # TAB 2: MODELING RESULTS (uses cached models)
    # =====================================================
    with tab_model:
        st.subheader("Modeling Results: Predicting Life Expectancy")

        try:
            artifacts = get_model_artifacts(imputed_df)
        except ValueError as e:
            st.error(str(e))
            return

        model_choice = st.radio(
            "Select model to view:",
            options=["Linear Regression", "Random Forest"],
            horizontal=True,
        )

        if model_choice == "Linear Regression":
            st.markdown(
                f"**Test R²:** {artifacts['r2_lr']:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; **Test RMSE:** {artifacts['rmse_lr']:.3f}"
            )

            coef_df = pd.DataFrame(
                {
                    "Variable": artifacts["X_columns"],
                    "Coefficient": artifacts["lr_coef"],
                }
            )
            coef_df["abs_coef"] = coef_df["Coefficient"].abs()
            coef_sorted = coef_df.sort_values("abs_coef", ascending=True)

            st.markdown("#### Coefficient Plot (standardized predictors)")
            plt.figure(figsize=(10, 8))
            colors = np.where(
                coef_sorted["Coefficient"] > 0, "skyblue", "salmon"
            )
            plt.barh(coef_sorted["Variable"], coef_sorted["Coefficient"], color=colors)
            plt.axvline(0, color="grey", linestyle="--", linewidth=1)
            plt.xlabel("Standardized Coefficient")
            plt.title("Linear Regression Coefficients")
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            st.pyplot(plt.gcf())

        else:  # Random Forest
            st.markdown(
                f"**Test R²:** {artifacts['r2_rf']:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; **Test RMSE:** {artifacts['rmse_rf']:.3f}"
            )

            imp_df = pd.DataFrame(
                {
                    "Variable": artifacts["X_columns"],
                    "Importance": artifacts["rf_importances"],
                }
            )
            imp_sorted = imp_df.sort_values("Importance", ascending=True)

            st.markdown("#### Feature Importance (Random Forest)")
            plt.figure(figsize=(10, 8))
            plt.barh(imp_sorted["Variable"], imp_sorted["Importance"])
            plt.xlabel("Importance")
            plt.title("Random Forest Feature Importance")
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            st.pyplot(plt.gcf())


# =========================================================
# Page 4: Modeling (LR + RF)
# =========================================================
def interactive_page():
    st.title("🎛️ Interactive Exploration")

    st.markdown(
        """
        This page is designed as a **playground** for exploring the data and models:

        - **Interactive EDA:** filter by region/country and explore distributions, relationships, and maps  
        - **Scenario Simulator:** adjust key variables and see how the **Linear Regression** model's
          predicted life expectancy changes  
        """
    )

    df = imputed_df.copy()

    tab_eda, tab_model = st.tabs(
        ["📊 Interactive EDA", "🧮 Scenario Simulator"]
    )

        # =====================================================
    # TAB 1: INTERACTIVE EDA
    # =====================================================
    with tab_eda:
        st.subheader("Interactive EDA")

        st.markdown(
            """
            Use the controls below to:
            - Filter by **region** or **country name**  
            - Explore **distributions** of any numeric variable  
            - See how variables relate to **life expectancy**  
            - View **global patterns** on a map  
            """
        )

        # -------------------------
        # Filters (moved from sidebar into this section)
        # -------------------------
        col_filter_left, col_filter_right = st.columns(2)

        if "region" in df.columns:
            regions = df["region"].dropna().unique()
            with col_filter_left:
                selected_regions = st.multiselect(
                    "Filter by Region:",
                    options=regions,
                    default=list(regions),
                )
        else:
            selected_regions = []

        with col_filter_right:
            country_search = st.text_input(
                "Search for a country (optional):",
                "",
            )

        filtered_df = df.copy()
        if selected_regions:
            filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]
        if country_search:
            filtered_df = filtered_df[
                filtered_df["country"].str.contains(
                    country_search, case=False, na=False
                )
            ]

        numeric_columns = filtered_df.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != "year"]

        t1, t2, t3, t4 = st.tabs(
            ["Distributions", "Scatter vs Life Expectancy", "Relationships", "Map"]
        )

        # ---- Distributions ----
        with t1:
            st.subheader("Distributions of Numeric Variables")
            if numeric_columns:
                default_var = (
                    "life_expectancy"
                    if "life_expectancy" in numeric_columns
                    else numeric_columns[0]
                )
                dist_var = st.selectbox(
                    "Select a numeric variable:",
                    options=numeric_columns,
                    index=numeric_columns.index(default_var),
                )

                hist_df = filtered_df.dropna(subset=[dist_var])
                hist_fig = px.histogram(
                    hist_df,
                    x=dist_var,
                    nbins=50,
                    title=f"Distribution of {dist_var.replace('_',' ').title()}",
                )
                st.plotly_chart(hist_fig, use_container_width=True)

                if "region" in filtered_df.columns:
                    box_df = filtered_df.dropna(subset=[dist_var])
                    box_fig = px.box(
                        box_df,
                        x="region",
                        y=dist_var,
                        color="region",
                        points="all",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        title=f"{dist_var.replace('_',' ').title()} by Region",
                    )
                    st.plotly_chart(box_fig, use_container_width=True)
            else:
                st.info("No numeric variables available for interactive distributions.")

        # ---- Scatter vs Life Expectancy ----
        with t2:
            st.subheader("Life Expectancy vs Other Variables")
            if "life_expectancy" in numeric_columns and len(numeric_columns) > 1:
                x_options = [c for c in numeric_columns if c != "life_expectancy"]
                default_x = (
                    "gdp_per_capita"
                    if "gdp_per_capita" in x_options
                    else x_options[0]
                )
                x_var = st.selectbox(
                    "X-axis variable:",
                    options=x_options,
                    index=x_options.index(default_x),
                )

                scatter_df = filtered_df.dropna(subset=[x_var, "life_expectancy"])
                if "population" in scatter_df.columns:
                    scatter_df["population"] = scatter_df["population"].fillna(1)
                    size_col = "population"
                else:
                    size_col = None

                scatter_fig = px.scatter(
                    scatter_df,
                    x=x_var,
                    y="life_expectancy",
                    color="region" if "region" in scatter_df.columns else None,
                    size=size_col,
                    size_max=35,
                    hover_name="country"
                    if "country" in scatter_df.columns
                    else None,
                    trendline="ols",
                    title=f"Life Expectancy vs {x_var.replace('_',' ').title()}",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                st.plotly_chart(scatter_fig, use_container_width=True)

                if len(scatter_df) > 2:
                    corr_val = scatter_df[x_var].corr(scatter_df["life_expectancy"])
                    st.markdown(f"**Correlation:** {corr_val:.2f}")
            else:
                st.info("Need life_expectancy and at least one other numeric variable.")

        # ---- Any Two Variables ----
        with t3:
            st.subheader("Relationships Between Any Two Variables")
            if len(numeric_columns) >= 2:
                x_any = st.selectbox("X-axis:", options=numeric_columns, index=0)
                y_any = st.selectbox("Y-axis:", options=numeric_columns, index=1)

                rel_df = filtered_df.dropna(subset=[x_any, y_any])
                if "population" in rel_df.columns:
                    rel_df["population"] = rel_df["population"].fillna(1)
                    size_col = "population"
                else:
                    size_col = None

                rel_fig = px.scatter(
                    rel_df,
                    x=x_any,
                    y=y_any,
                    color="region" if "region" in rel_df.columns else None,
                    size=size_col,
                    size_max=35,
                    hover_name="country"
                    if "country" in rel_df.columns
                    else None,
                    title=f"{y_any.replace('_',' ').title()} vs {x_any.replace('_',' ').title()}",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                st.plotly_chart(rel_fig, use_container_width=True)

                if len(rel_df) > 2:
                    corr_val = rel_df[x_any].corr(rel_df[y_any])
                    st.markdown(f"**Correlation:** {corr_val:.2f}")
            else:
                st.info("Need at least two numeric variables for this view.")

        # ---- Map ----
        with t4:
            st.subheader("Global Map of Selected Indicator")
            if "year" in filtered_df.columns and "country" in filtered_df.columns:
                map_var_options = [
                    v
                    for v in [
                        "life_expectancy",
                        "gen_gov_health_expenditure_pct_gdp",
                        "gdp_per_capita",
                        "infant_deaths",
                        "adult_mortality",
                    ]
                    if v in filtered_df.columns
                ]
                if map_var_options:
                    map_var = st.selectbox(
                        "Select variable to map:",
                        options=map_var_options,
                        index=0,
                    )
                    years = sorted(filtered_df["year"].dropna().unique())
                    if years:
                        selected_year = st.slider(
                            "Select Year",
                            min_value=int(min(years)),
                            max_value=int(max(years)),
                            value=int(min(years)),
                            step=1,
                        )
                        map_df = filtered_df[
                            filtered_df["year"] == selected_year
                        ].dropna(subset=[map_var])

                        min_val = filtered_df[map_var].min()
                        max_val = filtered_df[map_var].max()

                        map_fig = px.choropleth(
                            map_df,
                            locations="country",
                            locationmode="country names",
                            color=map_var,
                            hover_name="country",
                            hover_data={"year": True, map_var: True},
                            color_continuous_scale=px.colors.sequential.Viridis,
                            range_color=[min_val, max_val],
                            title=f"{map_var.replace('_',' ').title()} in {selected_year}",
                        )
                        st.plotly_chart(map_fig, use_container_width=True)
                else:
                    st.info("No suitable variables available for the map.")
            else:
                st.info("Need 'country' and 'year' columns to draw the map.")


    # =====================================================
    # TAB 2: SCENARIO SIMULATOR (INTERACTIVE MODELING)
    # =====================================================
    with tab_model:
        st.subheader("Scenario Simulator: Policy 'What-If' Tool")

        st.markdown(
            """
            This tool uses the **Linear Regression** model to explore *what might happen* to
            predicted life expectancy under different conditions.

            1. Choose a **baseline country and year**  
            2. Adjust key variables (e.g., GDP, health spending, safe water)  
            3. See how the model's **predicted life expectancy** changes  
            """
        )

        # Get cached model artifacts
        try:
            artifacts = get_model_artifacts(imputed_df)
        except ValueError as e:
            st.error(str(e))
            return

        lr = artifacts["lr"]
        scaler = artifacts["scaler"]
        X_cols = artifacts["X_columns"]
        lr_coef = artifacts["lr_coef"]

        # -------------------------
        # 1. Choose baseline row
        # -------------------------
        if "country" in df.columns and "year" in df.columns:
            countries = sorted(df["country"].dropna().unique().tolist())
            selected_country = st.selectbox("Baseline country:", options=countries)

            df_country = df[df["country"] == selected_country]
            years = sorted(df_country["year"].dropna().unique().tolist())
            selected_year = st.selectbox("Baseline year:", options=years)

            baseline_row = df_country[df_country["year"] == selected_year]
            if baseline_row.empty:
                st.warning("No data available for this country/year combination.")
                return
            baseline_row = baseline_row.iloc[0]
        else:
            st.error("`country` and `year` columns are required for the scenario simulator.")
            return

        # Build baseline feature vector in the same order as X_cols
        baseline_x = baseline_row[X_cols].copy()

        # Baseline model prediction (use DataFrame so scaler sees feature names)
        baseline_df_for_scaler = pd.DataFrame([baseline_x], columns=X_cols)
        baseline_z = scaler.transform(baseline_df_for_scaler)
        baseline_pred = float(lr.predict(baseline_z)[0])

        baseline_actual = (
            float(baseline_row["life_expectancy"])
            if "life_expectancy" in baseline_row.index
            else None
        )

        st.markdown(
            f"**Baseline (model prediction) for {selected_country}, {selected_year}:** "
            f"{baseline_pred:.2f} years"
        )
        if baseline_actual is not None:
            st.caption(f"Observed life expectancy in data: {baseline_actual:.2f} years")

        # Create a container that will appear *above* the sliders
        metrics_container = st.container()

        st.markdown("---")

        # -------------------------
        # 2. Sliders for key variables
        # -------------------------
        st.markdown("### Adjust key variables")

        # Pick a subset of interpretable variables, only if they exist
        candidate_vars = [
            "gdp_per_capita",
            "gen_gov_health_expenditure_pct_gdp",
            "gen_gov_health_expenditure_pct_gov_exp",
            "education_expenditure_pct_gdp",
            "access_to_safe_water_pct_pop",
            "infant_deaths",
            "adult_mortality",
        ]
        slider_vars = [v for v in candidate_vars if v in X_cols]

        if not slider_vars:
            st.info("No expected slider variables found in the feature set.")
            return

        new_x = baseline_x.copy()
        slider_values = {}

        col_left, col_right = st.columns(2)
        half = (len(slider_vars) + 1) // 2
        left_vars = slider_vars[:half]
        right_vars = slider_vars[half:]

        def make_slider(var_name, col_container):
            col_min = float(df[var_name].min())
            col_max = float(df[var_name].max())
            default = float(baseline_row[var_name])

            with col_container:
                value = st.slider(
                    label=var_name.replace("_", " ").title(),
                    min_value=float(col_min),
                    max_value=float(col_max),
                    value=default,
                )
            return value

        for var in left_vars:
            slider_values[var] = make_slider(var, col_left)
        for var in right_vars:
            slider_values[var] = make_slider(var, col_right)

        # Apply slider values to new_x
        for var, val in slider_values.items():
            new_x[var] = val

        # -------------------------
        # 3. New prediction (uses sliders)
        # -------------------------
        new_df_for_scaler = pd.DataFrame([new_x], columns=X_cols)
        new_z = scaler.transform(new_df_for_scaler)
        new_pred = float(lr.predict(new_z)[0])
        delta = new_pred - baseline_pred

        # Fill the metrics container: this will render ABOVE the sliders area
        with metrics_container:
            st.markdown("### Predicted Life Expectancy Under Scenario")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="Predicted life expectancy (scenario)",
                    value=f"{new_pred:.2f} years",
                    delta=f"{delta:+.2f} years vs. baseline",
                )
            with col_b:
                if baseline_actual is not None:
                    st.metric(
                        label="Scenario vs observed (data)",
                        value=f"{new_pred - baseline_actual:+.2f} years",
                    )

        st.markdown(
            """
            The change above reflects how the **Linear Regression** model responds to the
            adjustments made to the selected variables, holding all other features at their
            baseline values for this country and year.
            """
        )


# =========================================================
# Page 5: Importance & Conclusion
# =========================================================
def importance_conclusion():
    st.title("📌 Final Insights, Implications, and Future Directions")

    st.markdown(
        """
        This page brings together the major insights from the **data processing**, **exploratory analysis**, **predictive modeling**, and **interactive scenario tools**. 
        The goal is to summarize what the data reveals, what the models learned, and how these findings can inform real-world decisions about health investment and public resource allocation.
        """
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # KEY FINDINGS – EDA
    # ------------------------------------------------------------
    st.subheader("🔍 Key Findings from Exploratory Data Analysis")

    st.markdown(
        """
        The EDA highlighted several strong and consistent patterns across countries:

        **1. Mortality indicators dominate the life expectancy landscape**  
        - Countries with high *infant deaths*, *under–five mortality*, and *adult mortality* show sharply lower life expectancy.  
        - These relationships are strong, stable, and visible in both correlations and scatter plots.

        **2. Economic conditions and government investment matter**  
        - Higher **GDP per capita** and **government health expenditure** tend to align with higher life expectancy.  
        - Access to safe water and stronger health infrastructure (physicians, nurses, hospital beds) are also linked to better health outcomes.

        **3. PCA revealed broad global patterns**  
        - One major dimension reflects a **wealth & resources gradient**, where wealthier nations cluster with higher health spending, more infrastructure, and longer lives.  
        - A second dimension relates to **disease burden and environmental factors**, separating high-mortality, low-access regions from the rest.

        Overall, the EDA paints a consistent picture: health outcomes emerge from a combination of **economic capacity**, **public investment**, and **basic resource availability**.
        """
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # KEY FINDINGS – MODELING
    # ------------------------------------------------------------
    st.subheader("🤖 Key Findings from Predictive Modeling")

    st.markdown(
        """
        **1. Linear Regression**  
        - Provided clear, interpretable relationships.  
        - Mortality variables showed the largest negative coefficients.  
        - GDP per capita, safe water access, and government health spending showed positive associations with life expectancy.  
        - Some vaccination or infrastructure variables occasionally showed unexpected signs, likely due to **multicollinearity** rather than true negative effects.

        **2. Random Forest**  
        - Achieved strong predictive performance with high R² and low RMSE.  
        - Identified similar top predictors as Linear Regression, reinforcing the consistency of the findings: mortality indicators, GDP per capita, spending levels, and health system capacity.

        **3. Agreement between models**  
        - Both models point to the same core drivers of life expectancy.  
        - This convergence increases confidence in the results.
        """
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # INTERACTIVE INSIGHTS
    # ------------------------------------------------------------
    st.subheader("⚠️ Limitations")

    st.markdown(
        """
        While the analysis provides meaningful insights, several limitations should be kept in mind:

        **1. Observational data cannot establish causation**  
        - All relationships identified are **associational**, not causal.  
        - Factors like government spending, GDP, and health access may be influenced by deeper structural conditions not fully captured in the dataset.

        **2. Missing data and imputation introduce uncertainty**  
        - Some variables required substantial imputation (via MICE), which can smooth over true variability or introduce bias—especially for countries with limited reporting.  
        - Imputed values should be interpreted cautiously.

        **3. Multicollinearity affects interpretability**  
        - Many predictors are highly correlated (e.g., GDP, health spending, workforce capacity).  
        - Linear regression coefficients may not always reflect independent effects.

        **4. Inconsistent reporting across countries**  
        - Measurements vary in accuracy and reliability across sources and regions.  
        - Some indicators (e.g., water access, health workforce) are updated infrequently.

        **5. Models are not forecasts**  
        - The models describe patterns in historical data, but they do not necessarily predict future life expectancy without accounting for time dynamics or policy changes.

        Together, these limitations highlight the need for cautious interpretation and underscore that the results should serve as **contextual insights**, not definitive policy prescriptions.
        """
    )


    st.markdown("---")

    # ------------------------------------------------------------
    # REAL-WORLD IMPLICATIONS
    # ------------------------------------------------------------
    st.subheader("🌍 Real-World Implications")

    st.markdown(
        """
        This project highlights several important themes for global health policy:

        - **Investing in foundational public health resources—clean water, maternal & child health services, and primary care systems—has the biggest potential impact.**

        - **Economic development and health investment go hand-in-hand.** Countries with more fiscal capacity tend to sustain stronger health systems.

        - **Mortality reduction remains the clearest pathway to improving life expectancy**, especially reductions in infant, child, and adult mortality.

        - **Infrastructure and workforce capacity** (physicians, nurses, hospital beds) are essential to translating spending into real-world outcomes.

        These insights support the idea that improving life expectancy requires **both resources and efficient public investment**, not one or the other.
        """
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # FUTURE DIRECTIONS
    # ------------------------------------------------------------
    st.subheader("🚀 Future Directions")

    st.markdown(
        """
        Several promising directions could expand or strengthen this work:

        **1. Time-lagged modeling**  
        - Government spending in year *t* may influence outcomes in year *t+1* or later.  
        - Incorporating lags would better capture real-world policy effects.

        **2. Additional social determinants**  
        - Inequality, education quality, environmental exposure, and political stability could deepen the analysis.

        **3. Forecasting models**  
        - ARIMA, Prophet, or LSTM models could project future life expectancy under different investment scenarios.

        **4. Causal inference approaches**  
        - Techniques like DID, IV regression, or synthetic controls could help separate correlation from causation.

        **5. Country case studies**  
        - Deep dives into individual nations could reveal context-specific drivers not visible in global models.

        Together, these extensions would strengthen the policy relevance and explanatory power of the analysis.
        """
    )



# ---------------------------
# Page Routing
# ---------------------------
page_routes = {
    "Project Overview": project_overview,
    "Data Processing": data_processing,
    "EDA and Modeling": eda_model_results_page,
    "Interactive Page": interactive_page,
    "Importance & Conclusion": importance_conclusion
}

page_routes[selection]()
