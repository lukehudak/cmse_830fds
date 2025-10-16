import streamlit as st

st.title("Project Overview")

st.subheader("Goal")
st.write("Explore how government expenditures, particularly on health, relate to population health outcomes across countries.")

# -----------------------------
# Original Data Sources
# -----------------------------
st.subheader("Original Datasets & Sources")

st.markdown("""
The data was compiled from multiple publicly available datasets:  

1. **Health Metrics**  
   - Source: [Kaggle, Life Expectancy (WHO) Fixed](https://www.kaggle.com/datasets/lashagoch/life-expectancy-who-updated/data)  
   - Includes data such as life expectancy, infant mortality, adult mortality, immunization coverage, BMI, HIV incidents, and alcohol consumption for individual countries.
   - Data spans from 2000–2015.

2. **Government Expenditure Data**  
   - Source: [World Bank Open Data](https://data.worldbank.org/)  
   - Includes health, education, and R&D expenditure as a percentage of GDP.  
   - Data spans from 2000–2024.
""")

# -----------------------------
# Variable Descriptions
# -----------------------------
st.subheader("Variables")

st.markdown("""
- **Country**: Name of the country  
- **Region**: Region the country belongs to  
- **Year**: Year of observation  
- **Life expectancy**: Average life expectancy of both genders  
- **Infant deaths**: Infant deaths per 1000 population  
- **Under five deaths**: Deaths of children under five per 1000 population  
- **Adult mortality**: Adult deaths per 1000 population  
- **Alcohol consumption**: Liters of pure alcohol per capita (15+ years old)  
- **HepatitisB**: % coverage of HepB3 immunization among 1-year-olds  
- **Measles**: % coverage of MCV1 immunization among 1-year-olds  
- **BMI**: Body Mass Index (kg/m²)  
- **Polio**: % coverage of Pol3 immunization among 1-year-olds  
- **Diphtheria**: % coverage of DTP3 immunization among 1-year-olds  
- **Incidents HIV**: HIV incidents per 1000 population aged 15-49  
- **Health expenditure**: Public health expenditure as % of GDP  
- **Education expenditure**: Education expenditure as % of GDP  
- **R&D expenditure**: Research & development expenditure as % of GDP  
- **GDP per capita**: Gross domestic product per capita (constant 2015 USD)  
- **Population**: Total population (midyear estimates)  
""")

# -----------------------------
# Data Processing Overview
# -----------------------------
st.subheader("Data Processing Overview")

st.write("""
This section summarizes the steps taken to go from the raw datasets to the final cleaned and merged DataFrame used in this analysis.
""")

st.subheader("Step 1: Load Raw Data")
st.markdown("""
- Health metrics data (life expectancy, mortality rates, vaccination coverage, BMI, alcohol consumption, HIV incidents, etc.) from **Life Expectancy (WHO) Fixed (Kaggle)**.
- Government expenditure and economic indicators (health, education, R&D expenditure, GDP per capita, population) from **World Bank Open Data**.
""")

st.subheader("Step 2: Clean and Preprocess")
st.markdown("""
- Unwanted columns were dropped from datasets and columns were renamed.
- Initial exploratory data analysis (EDA) was conducted to examine characteristics of the data.
""")

st.subheader("Step 3: Merge Datasets")
st.markdown("""
- Datasets were merged using **country and year** as keys.
- A **left merge** was used to retain as many observations as possible.
- Missingness in the merged dataset was visualized.
""")

st.subheader("Step 4: Handle Missing Values")
st.markdown("""
- Explored missingness patterns (MCAR, MAR, MNAR) using **heatmaps and proportions**.
- Missing values in `health_expenditure_pct_gdp` were imputed using **stochastic regression** with correlated numeric variables.
- After imputation, all rows used for analysis were complete for numeric columns.
""")

st.subheader("Step 5: Encode Categorical Variables")
st.markdown("""
- `region` column was **one-hot encoded** for visualization and analysis.
- Missing `region` values were filled with "Unknown".
- Original DataFrame was preserved, and encoded columns were added to create `encoded_df`.
""")

st.subheader("Step 6: Final Cleaned DataFrame")
st.markdown("""
- Final DataFrame (`encoded_df`) contains all numerical columns ready for visualization and analysis.
- No missing values remain in columns used for plots.
- The DataFrame is saved as `encoded_df.csv` for reproducibility.
""")

st.subheader("Workflow Summary Diagram (Optional)")
st.markdown("""
1. **Load raw datasets** → 2. **Clean and Preprocess** → 3. **Merge datasets** → 4. **Handle missing values** → 5. **Encode categorical variables** → 6. **Final cleaned DataFrame**
""")
