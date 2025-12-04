# final_streamlit_app.py
# CMSE 830 Midsemester Project Streamlit App
# Generated with the help of ChatGPT (version 5-mini) 2025-10-19

import streamlit as st
st.set_page_config(page_title="Health Data Project", layout="wide")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import plotly.express as px

# ---------------------------
# Load Data
# ---------------------------
#life_exp_df = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/Life-Expectancy-Data-Updated.csv")
#gov_metrics_df = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/World_Bank_data.csv")
#encoded_df = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/encoded_df.csv")
#merged_df = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/merged_df.csv")
#final_merged_df =pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/final_meged_df.csv")
#df_imputed =pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/df_imputed.csv")


#life_exp_df = pd.read_csv("data/Life-Expectancy-Data-Updated.csv")
#gov_metrics_df = pd.read_csv("data/World_Bank_data.csv")
#encoded_df = pd.read_csv("data/encoded_df.csv")
#merged_df = pd.read_csv("data/merged_df.csv")

life_exp_url = "https://raw.githubusercontent.com/lukehudak/cmse_830fds/main/data/Life-Expectancy-Data-Updated.csv"
gov_metrics_url = "https://raw.githubusercontent.com/lukehudak/cmse_830fds/main/data/World_Bank_data.csv"
encoded_df_url = "https://raw.githubusercontent.com/lukehudak/cmse_830fds/main/data/encoded_df.csv"
merged_df_url = "https://raw.githubusercontent.com/lukehudak/cmse_830fds/main/data/merged_df.csv"
final_merged_df_url= "https://raw.githubusercontent.com/lukehudak/cmse_830fds/main/data/final_merged_df.csv"
df_imputed_url = "https://raw.githubusercontent.com/lukehudak/cmse_830fds/main/data/df_imputed.csv"

#Load CSVs from GitHub
life_exp_df = pd.read_csv(life_exp_url)
gov_metrics_df = pd.read_csv(gov_metrics_url)
encoded_df = pd.read_csv(encoded_df_url)
merged_df = pd.read_csv(merged_df_url)
final_merged_df = pd.read_csv(final_merged_df_url)
df_imputed = pd.read_csv(df_imputed_url)

# Make copies of imputed and median imputed dfs if needed
imputed_df = encoded_df.copy()
median_imputed_df = imputed_df.copy()  # If needed for visualization comparison

# ---------------------------
# Page Setup
# ---------------------------
#st.set_page_config(page_title="Health Data Project", layout="wide")
st.sidebar.title("Navigation")
pages = [
    "Project Overview",
    "Data Collection & Preparation",
    "Data Processing",
    "Exploratory Data Analysis",
    "Interactive Analysis",
    "Modeling",  # <-- interactive modeling page
    "Summary / Conclusion"
]
selection = st.sidebar.radio("Select Page:", pages)

# ---------------------------
# Page 1: Project Overview
# ---------------------------
def project_overview():
    st.title("🌍 Health Data Project Overview")
    st.subheader("By: Luke Hudak")
    st.markdown("""
    **Summary:**  
    This project examines the relationship between government expenditures and population health across countries and years.  
    Each page in this app shows a step in the workflow, explaining what was done, why it was done, and what insights can be derived.
    """)

    # Use columns for visually appealing step summary
    st.subheader("Project Workflow")
    col1, col2, col3 = st.columns(3)

    col1.markdown("### 1️⃣ Data Collection")
    col1.markdown("""
    - Sources: Kaggle (Life Expectancy), World Bank (Gov Expenditure)
    - Load raw datasets
    - Inspect structure and missingness
    """)

    col2.markdown("### 2️⃣ Data Processing")
    col2.markdown("""
    - Handle missing values using stochastic regression & median imputation
    - Encode categorical variables (one-hot encoding for 'region')
    - Scale numeric data for analysis
    """)

    col3.markdown("### 3️⃣ Analysis & EDA")
    col3.markdown("""
    - Explore distributions, correlations, and relationships
    - Perform **Principal Component Analysis (PCA)** to reduce dimensionality and summarize key patterns
    - Prepare data for interactive visualization and modeling
    """)

    st.markdown("---")
    st.subheader("Predictive Modeling")
    st.markdown("""
    - Built **Linear Regression** and **Random Forest** models to predict life expectancy
    - Linear Regression highlights linear effects of key variables (physicians, BMI, hospital beds, government health expenditure)
    - Random Forest identifies non-linear interactions, with under-five deaths and adult mortality as strongest predictors
    - Model insights inform understanding of which factors most influence population health
    """)

    st.markdown("---")
    st.subheader("Interactive Exploration")
    st.markdown("""
    - Users can select regions and variables to generate interactive visualizations and explore trends, distributions, and relationships in the dataset.
    """)
    
    st.subheader("Summary & Recommendations")
    st.markdown("""
    - Key Findings: Life expectancy positively related to government expenditures, mortality indicators critical
    - Next Steps: Extend modeling, explore additional health/socioeconomic indicators
    - Importance: Insights support evidence-based policy decisions for improving population health
    """)



# ---------------------------
# Page 2: Data Collection & Preparation (IDA)
# ---------------------------
def data_collection_preparation():
    st.title("📥 Data Collection & Preparation")
    
    st.markdown("""
    **Page Summary:**  
    This page details how and where the raw datasets were collected, cleaned, and merged into a single dataset used for analysis. 
    The steps are as follows:
    """)

    # ---------------------------
    # Section 1: Raw Life Expectancy Dataset
    # ---------------------------
    with st.expander("🔹 Life Expectancy Dataset"):
        st.markdown("""
        **Purpose:** The life expectancy dataset provides the primary outcome variable (`life_expectancy`) along with other health indicators (infant deaths, adult mortality, BMI, etc.) for each country over multiple years.  

        **What was done:**  
        - Loaded the dataset from Kaggle (link on overview page)
        - Removed unwanted columns (thinness metrics, schooling metrics, economy status indicators, etc.) to focus on relevant health indicators  
        - Checked data types, missing values, and duplicates  
        """)
        st.dataframe(life_exp_df.head())
        st.markdown("""
        **Outcome:** This process resulted in a properly formatted and cleaned dataset that countained all relevant health information needed for analysis.""")

    # ---------------------------
    # Section 2: Government Expenditure Dataset
    # ---------------------------
    with st.expander("🔹 Government Expenditure Dataset"):
        st.markdown("""
        **Purpose:** The government expenditure dataset provides data on government expenditure, along with GDP and population metrics for multiple countries over the span of multiple years.

        **What was done:**  
        - Loaded data from the World Bank Open Data (link on overview page)
        - Dropped unnecessary columns (`Time Code`) and cleaned unwanted rows  
        - Renamed columns for clarity (e.g., `Health_expenditure (% of GDP)` → `health_expenditure_pct_gdp`)  
        - Converted columns to appropriate data types  
        - Replaced missing or placeholder values (`..`) with `NaN`  
        """)
        st.dataframe(gov_metrics_df.head())
        st.markdown("""
          **Outcome:** This process resulted in a properly formatted and cleaned dataset that provides the explanatory variables needed to assess how government spending relates to health outcomes.
        """)

    # ---------------------------
    # Section 3: Merging Datasets
    # ---------------------------
    with st.expander("🔹 Merging Datasets"):
        st.markdown("""
        **Purpose:** To combine the life expectancy and government expenditure datasets into a single dataframe to be used in analysis.

        **What was done:**  
        - Merged datasets using a **left join** on `Country` and `Year` to preserve all life expectancy records  
        - Standardized column names to maintain consistency  
        - Validated merge success by checking for missing values, duplicates, and alignment across key identifiers  
        """)
        st.dataframe(merged_df.head())
        st.markdown("""
        **Outcome:** The merged dataset combines health outcomes and spending indicators across countries and years, resulting in a single dataframe used for analysis.
        """)
    
        # ---------------------------
    # Section 4: Variable Descriptions
    # ---------------------------
    with st.expander("🔹 Description of Variables in Final Merged Dataset"):
        st.markdown("""
        | **Variable** | **Description** | **Type** |
        |---------------|-----------------|-----------|
        | `country` | Country name | *Categorical* |
        | `region` | Geographic region | *Categorical* |
        | `year` | Year of observation | *Integer* |
        | `infant_deaths` | Infant deaths per 1000 population | *Numeric* |
        | `under_five_deaths` | Deaths of children under five per 1000 population | *Numeric* |
        | `adult_mortality` | Adult deaths per 1000 population | *Numeric* |
        | `alcohol_consumption` | Liters of pure alcohol per capita (15+ years old) | *Numeric* |
        | `hepatitis_b` | % coverage of Hepatitis B (HepB3) immunization among 1-year-olds | *Numeric* |
        | `measles` | % coverage of Measles (MCV1) immunization among 1-year-olds | *Numeric* |
        | `bmi` | Body Mass Index (kg/m²) | *Numeric* |
        | `polio` | % coverage of Polio (Pol3) immunization among 1-year-olds | *Numeric* |
        | `diphtheria` | % coverage of Diphtheria, tetanus toxoid, and pertussis (DTP3) immunization among 1-year-olds | *Numeric* |
        | `incidents_hiv` | HIV incidents per 1000 population aged 15–49 | *Numeric* |
        | `life_expectancy` | Average life expectancy of both genders (2000–2015) | *Numeric* |
        | `country_code` | Three-letter ISO country code | *Categorical* |
        | `health_expenditure_pct_gdp` | Public health expenditure as % of GDP | *Numeric* |
        | `education_expenditure_pct_gdp` | Public education expenditure as % of GDP | *Numeric* |
        | `rnd_expenditure_pct_gdp` | Gross domestic R&D expenditure as % of GDP | *Numeric* |
        | `gdp_per_capita` | GDP per capita (constant 2015 USD) | *Numeric* |
        | `population` | Midyear total population estimate | *Numeric* |
        """)


    # ---------------------------
    # Section 5: Summary Statistics
    # ---------------------------
    with st.expander("🔹 Summary Statistics"):
        st.markdown("""
        **Purpose:** To understand the data distribution, detect anomalies, and identify the range of values for key metrics before performing analysis.

        **What was done:**  
        - Selected essential numeric variables (life expectancy, government expenditure, infant deaths, adult mortality, GDP per capita)  
        - Computed descriptive statistics including mean, median, min, max, and standard deviation  
        """)
        key_vars = ["life_expectancy","health_expenditure_pct_gdp","infant_deaths","adult_mortality","education_expenditure_pct_gdp","gdp_per_capita"]
        st.dataframe(imputed_df[key_vars].describe().T)
        st.markdown("""
        **Outcome:** Summary statistics provided an initial understanding of the scale, variability, and potential outliers within key variables.
        """)

# ---------------------------
# Page 3: Data Processing
# ---------------------------
def data_processing():
    st.title("⚙️ Data Processing")
    global merged_df, imputed_df, median_imputed_df

    st.markdown("""
    **Page Summary:**  
    This page describes the steps taken to process the dataset, focusing on handling missing values and encoding categorical variables.  
    These steps ensure that the data is accurate, model-ready, and interpretable for subsequent analyses.
    """)

    # ---------------------------
    # Section 1: Missing Values Overview
    # ---------------------------
    with st.expander("🔹 Missing Values Overview and Visualization"):
        st.markdown("""
        **Purpose:** To identify where and how much data is missing across variables.

        **What was done:**  
        - Computed total and percentage of missing values per variable (used to assess which columns to impute and/or drop)
        - Created a heatmap of missing values using Seaborn  
        - Each row represents an observation, and each column represents a variable  
        - Missing values are highlighted visually for quick detection  
        """)
        st.dataframe(merged_df.isna().sum())
        fig, ax = plt.subplots(figsize=(12,6))
        sns.heatmap(merged_df.isna().T, cmap="magma")
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Columns")
        st.pyplot(fig)
        st.markdown("""
        **Outcome:** Better understanding of the missingness in the merged dataframe. Visualization allowed for better understanding of scope of missingness in main predictor variable (health expenditure).
        """)



    # ---------------------------
    # Section 2: Imputation Methods
    # ---------------------------
    with st.expander("🔹 Imputation: Stochastic Regression vs Median"):
        st.markdown("""
        **Purpose:** To use two different methods to impute missing values for `health_expenditure_pct_gdp`, and decide which is most suitable for the project.

        **What was done:**  
        - **Stochastic Regression:** Predicted missing values for health expenditure using correlated variables (`life_expectancy` and `infant_deaths`) while adding synthetic noise to simulate random variablilty
        - **Median Imputation:** Replaced missing values for health expenditure with the median value of the column
        - Compared both imputation methods using a scatter plot to visualize
        """)

        median_imputed_df = merged_df.copy()
        median_value = median_imputed_df['health_expenditure_pct_gdp'].median()

        median_imputed_df['health_expenditure_pct_gdp'].fillna(median_value, inplace=True)
        missing_mask = merged_df['health_expenditure_pct_gdp'].isna()  # missing value mask
        observed_mask = ~missing_mask  # observed mask

        # x axis
        x_observed = merged_df.loc[observed_mask, 'life_expectancy']
        x_missing = merged_df.loc[missing_mask, 'life_expectancy']

        # y axis
        y_observed = merged_df.loc[observed_mask, 'health_expenditure_pct_gdp']
        y_stochastic = imputed_df.loc[missing_mask, 'health_expenditure_pct_gdp']
        y_median = median_imputed_df.loc[missing_mask, 'health_expenditure_pct_gdp']

        # plot
        fig, ax = plt.subplots(figsize=(9,6))
        ax.scatter(x_observed, y_observed, label='Observed', color='black', alpha=0.2)  # observed vals
        ax.scatter(x_missing, y_stochastic, label='Stochastic Regression Imputed', color='red', alpha=0.7)  # stochastic vals
        ax.scatter(x_missing, y_median, label='Median Imputed', color='blue', alpha=0.7)  # median vals
        ax.set_xlabel('Life Expectancy')
        ax.set_ylabel('Health Expenditure (% GDP)')
        ax.set_title('Comparison of Imputation Methods for Missing Values')
        ax.legend()
        ax.grid(alpha=0.4)

        # --- Streamlit rendering ---
        st.pyplot(fig)
        st.markdown("""
        **Outcome:** Stochastic regression imputation produced more realistic values (compared to median imputation) that align with observed data trends.
        """)

    # ---------------------------
    # Section 4: Encoding Categorical Variables
    # ---------------------------
    with st.expander("🔹 Encoding: Region Variable"):
        st.markdown("""
        **Purpose:** To prepare the categorical variable 'region' for statistical analysis converting the variable into numerical format using one-hot encoding.

        **What was done:**  
        - Filled missing region values with "Unknown"  
        - Applied **One-Hot Encoding** to create binary columns for each region  
        - Combined encoded columns with the existing dataset  
        """)
        region_cols = [col for col in encoded_df.columns if col not in imputed_df.columns]
        st.dataframe(encoded_df.head())
        st.markdown("""
        **Outcome:** A dataframe with 'region' encoded that will allow for simplified region-specific analysis (scroll to the right to view encoding).
        """)


# ---------------------------
# PAGE: Exploratory Data Analysis (EDA)
# ---------------------------
def eda():
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.markdown("""
    The goal of this section is to **understand the structure, distribution, and relationships** within the dataset.  
    Each section below provides descriptive and visual insights into different aspects of the data.  
    """)

    # =====================================================
    # Section 1: Summary Statistics
    # =====================================================
    with st.expander("🔹 Summary Statistics"):
        st.markdown("""
        **Purpose:**  
        Summary statistics provide a quick numerical overview of the dataset using mean, median, standard deviation, etc.
        """)
        
        key_vars = [
            "life_expectancy", "health_expenditure_pct_gdp", 
            "education_expenditure_pct_gdp", "infant_deaths",
            "adult_mortality", "gdp_per_capita", "bmi", "alcohol_consumption"
        ]

        st.dataframe(imputed_df[key_vars].describe().T)
        
 

    # =====================================================
    # Section 2: Data Distribution (Univariate)
    # =====================================================
    with st.expander("🔹 Data Distribution (Univariate Analysis)"):
        st.markdown("""
        **Purpose:**  
        To analyze variable distribution.
        """)

        numeric_cols = [
            "life_expectancy", "infant_deaths", "adult_mortality", 
            "health_expenditure_pct_gdp", "education_expenditure_pct_gdp", 
            "rnd_expenditure_pct_gdp", "gdp_per_capita", "bmi", "alcohol_consumption"
        ]

        # Histograms
        st.subheader("Histograms of Numeric Variables")
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols[:len(axes)]):
            sns.histplot(imputed_df[col], bins=20, kde=True, ax=axes[i])
            axes[i].set_title(col.replace("_", " ").title())
        plt.tight_layout()
        st.pyplot(fig)

        # Simple Boxplots (non-interactive)
        st.subheader("Box Plots by Region")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.boxplot(x="region", y="life_expectancy", data=imputed_df, ax=axes[0])
        sns.boxplot(x="region", y="health_expenditure_pct_gdp", data=imputed_df, ax=axes[1])
        axes[0].set_title("Life Expectancy by Region")
        axes[1].set_title("Health Expenditure (% GDP) by Region")
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **Interpretation:**  
        - Africa has a wide range of life expectancy.
        - Both Asia and Central America and Caribbean show quite a few outliers that are above average in health expenditure.
        """)

   # =====================================================
    # Section 3: Variable Relationships
    # =====================================================
    with st.expander("🔹 Variable Relationships"):
        st.markdown("""
        **Purpose:**  
        To explore how variables interact (specifically how life expectancy relates to government expenditure).
        """)

        pair_vars = [
            "life_expectancy", "health_expenditure_pct_gdp", 
            "education_expenditure_pct_gdp", "infant_deaths", 
            "gdp_per_capita", "rnd_expenditure_pct_gdp"
        ]

        # Create pairplot
        fig = sns.pairplot(
            imputed_df[pair_vars + ["region"]],
            hue="region",
            corner=True,
            diag_kind="kde",
            height=6,
            aspect=1.5,
            plot_kws={"alpha":0.7, "s":50}
        )

        # Adjust font sizes for readability
        fig.fig.subplots_adjust(top=0.95)
        for ax in fig.axes.flatten():
            if ax is not None:
                ax.tick_params(labelsize=14)
                ax.xaxis.label.set_size(16)
                ax.yaxis.label.set_size(16)

        if fig._legend is not None:
            for text in fig._legend.get_texts():
                text.set_fontsize(14)

        # Render in Streamlit
        st.pyplot(fig.fig)

        st.markdown("""
        **Interpretation:**  
        - Life expectancy appears to increase as government expenditure increases.  
        - Infant death has a strong negative relationship with life expectancy.  
        """)


    # =====================================================
    # Section 4: Correlation Heatmap (Simplified)
    # =====================================================
    with st.expander("🔹 Correlation Heatmap"):
        st.markdown("""
        **Purpose:**  
        To measure the correlation between key numeric variables.  
        """)

        corr_vars = [
            "life_expectancy", "infant_deaths", "adult_mortality", 
            "health_expenditure_pct_gdp", "education_expenditure_pct_gdp", 
            "gdp_per_capita", "bmi", "rnd_expenditure_pct_gdp"
        ]

        corr = imputed_df[corr_vars].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                    square=True, cbar=True, linewidths=0.5, annot_kws={"size": 9})
        ax.set_title("Correlation Heatmap (Key Variables)", fontsize=14)
        st.pyplot(fig)

        st.markdown("""
        **Interpretation:**  
        - Life expectancy has a mild positive correlation with health expenditure, GDP, BMI, and R&D expenditure.
        - Infant death and adult mortality have strong negative correlations with life expectancy. 
        - Education expenditure does not show any real significant correlations with any of the other variables (besides potentially health expenditure).
        """)


# ---------------------------
# Page 5: Interactive Analysis
# ---------------------------
def interactive_analysis():
    import plotly.express as px

    st.title("Interactive Analysis")
    st.markdown("""
    Use the interactive tabs below to explore relationships, distributions, trends, and global patterns among the variables in the dataset used for this project.
                
    **Note**: 
    - Use the selection panel on the left side of the app (below the navigation panel) to filter between regions and even specific countries.
    - Hover over visualizations for more info such as specific country and numerical values.
    """)

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    regions = encoded_df['region'].dropna().unique()
    selected_regions = st.sidebar.multiselect("Select Regions:", options=regions, default=list(regions))

    country_search = st.sidebar.text_input("Search for a country (partial name):", "")

    # Filter dataframe
    filtered_df = encoded_df.copy()
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    if country_search:
        filtered_df = filtered_df[filtered_df['country'].str.contains(country_search, case=False, na=False)]

    numeric_columns = filtered_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in ["year"]]

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3, tab4= st.tabs([
        "Distributions", "Scatter vs Life Expectancy", "Variable Relationships", "Map"
    ])

    # -----------------------------
    # Tab 1: Distributions
    # -----------------------------
    with tab1:
        st.subheader("Distributions of Numeric Variables")

        dist_var = st.selectbox("Select a numeric variable:", options=numeric_columns, index=numeric_columns.index("life_expectancy"))

        hist_df = filtered_df.dropna(subset=[dist_var])
        hist_fig = px.histogram(
            hist_df,
            x=dist_var,
            nbins=50,
            title=f"Distribution of {dist_var.replace('_',' ').title()}"
        )
        hist_fig.update_traces(opacity=0.8)
        st.plotly_chart(hist_fig, use_container_width=True)

        box_df = filtered_df.dropna(subset=[dist_var])
        box_fig = px.box(
            box_df,
            x="region",
            y=dist_var,
            color="region",
            points="all",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title=f"{dist_var.replace('_',' ').title()} Boxplot by Region"
        )
        box_fig.update_traces(marker=dict(line_width=0), opacity=0.8)
        st.plotly_chart(box_fig, use_container_width=True)

    # -----------------------------
    # Tab 2: Scatter vs Life Expectancy
    # -----------------------------
    with tab2:
        st.subheader("Life Expectancy vs Other Variables")
        st.markdown("""
        Note: Filter out regions to reduce congestion of scatter plot
        """)

        x_var = st.selectbox(
            "Select variable for x-axis:",
            options=[col for col in numeric_columns if col != "life_expectancy"],
            index=[i for i,col in enumerate(numeric_columns) if col=="gdp_per_capita"][0]
        )

        scatter_df = filtered_df.dropna(subset=[x_var, "life_expectancy", "population"])
        scatter_df["population"] = scatter_df["population"].fillna(1)

        scatter_fig = px.scatter(
            scatter_df,
            x=x_var,
            y="life_expectancy",
            color="region",
            size="population",
            size_max=35,
            hover_name="country",
            trendline="ols",
            title=f"Life Expectancy vs {x_var.replace('_',' ').title()}",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        scatter_fig.update_traces(marker=dict(line_width=0), opacity=0.85)
        st.plotly_chart(scatter_fig, use_container_width=True)

        if len(scatter_df) > 2:
            corr_val = scatter_df[x_var].corr(scatter_df["life_expectancy"])
            st.markdown(f"**Correlation:** {corr_val:.2f}")

    # -----------------------------
    # Tab 3: Variable Relationships
    # -----------------------------
    with tab3:
        st.subheader("Relationships Between Variables")
        st.markdown("""
        Note: Filter out regions to reduce congestion of scatter plot
        """)

        x_any = st.selectbox("X-axis variable:", options=numeric_columns, index=0)
        y_any = st.selectbox("Y-axis variable:", options=numeric_columns, index=1)

        rel_df = filtered_df.dropna(subset=[x_any, y_any, "population"])
        rel_df["population"] = rel_df["population"].fillna(1)

        rel_fig = px.scatter(
            rel_df,
            x=x_any,
            y=y_any,
            color="region",
            size="population",
            size_max=35,
            hover_name="country",
            title=f"{y_any.replace('_',' ').title()} vs {x_any.replace('_',' ').title()}",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        rel_fig.update_traces(marker=dict(line_width=0), opacity=0.85)
        st.plotly_chart(rel_fig, use_container_width=True)

        if len(rel_df) > 2:
            corr_val = rel_df[x_any].corr(rel_df[y_any])
            st.markdown(f"**Correlation:** {corr_val:.2f}")

    # -----------------------------
    # Tab 4: Map
    # -----------------------------
    with tab4:
        st.subheader("Global Health Indicator Map")
        st.markdown("""
        Notes:
        - The slider below selects a single year, not a range of years
        - Hover over the may to view the actual variable value associated
        """)

        map_var_options = [
            "life_expectancy", 
            "health_expenditure_pct_gdp", 
            "gdp_per_capita", 
            "infant_deaths", 
            "adult_mortality"
        ]
        default_index = map_var_options.index("life_expectancy")
        map_var = st.selectbox(
            "Select variable to display",
            options=map_var_options,
            index=default_index
        )

        years = sorted(filtered_df['year'].dropna().unique())
        selected_year = st.slider(
            "Select Year",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=int(min(years)),
            step=1
        )

        map_df = filtered_df[filtered_df['year'] == selected_year].dropna(subset=[map_var])

        min_val = filtered_df[map_var].min()
        max_val = filtered_df[map_var].max()

        map_fig = px.choropleth(
            map_df,
            locations="country",
            locationmode="country names",
            color=map_var,
            hover_name="country",
            hover_data={ "year": True, map_var: True },
            color_continuous_scale=px.colors.sequential.Viridis,
            range_color=[min_val, max_val],
            title=f"{map_var.replace('_',' ').title()} Across Countries ({selected_year})"
        )
        st.plotly_chart(map_fig, use_container_width=True)

# ---------------------------
# Page 7: Interactive Modeling
# ---------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------
# Page 7: Predictive Modeling
# ---------------------------
def modeling_page():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    st.title("🤖 Predictive Modeling of Life Expectancy")
    st.markdown("""
    Explore predictive models for life expectancy using numeric government expenditure and health indicators.
    """)

    # --- 1. Define predictors and target ---
    X = df_imputed.drop(columns=['life_expectancy'])
    y = df_imputed['life_expectancy']

    # Select only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    # --- 2. Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 3. Model selection ---
    model_choice = st.radio(
        "Select model to visualize:",
        options=["Linear Regression", "Random Forest"]
    )

    if model_choice == "Linear Regression":
        # --- Linear Regression ---
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.subheader("Linear Regression Results")
        st.markdown(f"**R²:** {r2:.3f}  |  **RMSE:** {rmse:.3f}")

        # --- Coefficients ---
        coefficients = pd.DataFrame({
            'Variable': X.columns,
            'Coefficient': lr.coef_
        })
        coefficients["abs_coef"] = coefficients["Coefficient"].abs()
        coefficients_sorted = coefficients.sort_values("abs_coef", ascending=True)

        # --- Plot ---
        plt.figure(figsize=(12,8))
        colors = np.where(coefficients_sorted["Coefficient"] > 0, 'skyblue', 'salmon')
        plt.barh(coefficients_sorted["Variable"], coefficients_sorted["Coefficient"], color=colors)
        plt.axvline(0, color='grey', linestyle='--', linewidth=1)
        plt.xlabel("Coefficient")
        plt.title("Linear Regression Coefficients")
        plt.tight_layout()
        st.pyplot(plt.gcf())

    else:
        # --- Random Forest ---
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.subheader("Random Forest Regression Results")
        st.markdown(f"**R²:** {r2:.3f}  |  **RMSE:** {rmse:.3f}")

        # --- Feature importances ---
        importances = pd.DataFrame({
            'Variable': X.columns,
            'Importance': rf.feature_importances_
        })
        importances_sorted = importances.sort_values("Importance", ascending=True)

        plt.figure(figsize=(12,8))
        plt.barh(importances_sorted["Variable"], importances_sorted["Importance"], color='skyblue')
        plt.xlabel("Importance")
        plt.xscale('log')
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        st.pyplot(plt.gcf())



# ---------------------------
# Page 6: Summary / Conclusion
# ---------------------------
# ---------------------------
# Page 6: Summary / Conclusion
# ---------------------------
def summary_conclusion():
    st.title("Summary & Modeling Insights")
    st.markdown("This page summarizes key findings from the project, including predictive modeling results and actionable insights.")

    # ---------------------------
    st.subheader("Key Findings")
    st.markdown("""
        - **Life expectancy shows moderate positive relationships with government expenditure** (health, education, and R&D).  
        - Regions differ greatly in health expenditure and health outcomes.  
        - **Stochastic regression imputation** was more effective than median imputation for missing values.
        - **Linear Regression** identified physician density, BMI, hospital beds, and government health expenditure as significant predictors of life expectancy.
        - **Random Forest** highlighted **under-five deaths** and **adult mortality** as the most important predictors, indicating that observed mortality rates dominate predictions when non-linear interactions are considered.
    """)

    # ---------------------------
    st.subheader("Model Performance")
    st.markdown("""
        **Linear Regression:**  
        - R² = 0.983, RMSE = 1.502  
        - Captures strong linear relationships between predictors and life expectancy.

        **Random Forest:**  
        - Emphasizes the predictive power of mortality indicators over direct expenditure metrics.
    """)

    # ---------------------------
    st.subheader("Next Steps")
    st.markdown("""
        - Further examine country-level trends in the data.  
        - Add more health and socio-economic indicators to assess government impact on citizen health.  
        - Explore additional predictive modeling approaches and interactions.
    """)

    # ---------------------------
    st.subheader("Importance")
    st.markdown("""
        - Understanding how government investment in health, education, and R&D affects population health can guide policy decisions.  
        - Mortality and health system indicators both play key roles in shaping life expectancy outcomes.  
        - Insights from both linear and non-linear models highlight areas where interventions may be most effective.
    """)



# ---------------------------
# Page Routing
# ---------------------------
page_routes = {
    "Project Overview": project_overview,
    "Data Collection & Preparation": data_collection_preparation,
    "Data Processing": data_processing,
    "Exploratory Data Analysis": eda,
    "Interactive Analysis": interactive_analysis,
    "Modeling": modeling_page,
    "Summary / Conclusion": summary_conclusion
}

page_routes[selection]()
