
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Load Data
# -----------------------------
encoded_df = pd.read_csv("/Users/lukehudak/MSU/FS25/CMSE 830/Midsemester Project/encoded_df.csv")
encoded_df["year"] = pd.to_numeric(encoded_df["year"], errors="coerce")

# Map one-hot region columns to readable names
region_cols = [col for col in encoded_df.columns if col.startswith("region_")]
region_map = {col: col.replace("region_", "") for col in region_cols}
encoded_df["region_name"] = ""
for col, name in region_map.items():
    encoded_df.loc[encoded_df[col] == 1, "region_name"] = name

# -----------------------------
# App Header
# -----------------------------
st.title("Global Health & Expenditure Dashboard")
st.write("""
Explore health, economic, and demographic indicators across countries and regions.
""")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Global Filters")

# Use actual region names
regions = encoded_df["region"].dropna().unique()  # "region" column from your original df
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=regions,
    default=list(regions)
)

# Country search filter (text input)
country_search = st.sidebar.text_input("Search for a specific country (partial name):", "")

# -----------------------------
# Filter DataFrame based on sidebar inputs
# -----------------------------
filtered_df = encoded_df.copy()
if selected_regions:
    filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]
if country_search:
    filtered_df = filtered_df[filtered_df["country"].str.contains(country_search, case=False, na=False)]


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Key Metrics", "Distributions", "Trends", "Relationships", "Map"
])

numeric_columns = filtered_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
numeric_columns = [col for col in numeric_columns if col not in ["year"]]

# -----------------------------
# Tab 1: Key Metrics
# -----------------------------
with tab1:
    st.subheader("Key Metrics for Selected Filters")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Life Expectancy", f"{filtered_df['life_expectancy'].mean():.1f}")
    col2.metric("Median Health Expenditure (% GDP)", f"{filtered_df['health_expenditure_pct_gdp'].median():.2f}")
    col3.metric("Median GDP per Capita (USD)", f"{filtered_df['gdp_per_capita'].median():,.0f}")

# -----------------------------
# Tab 2: Distributions
# -----------------------------
with tab2:
    st.subheader("Distributions of Numeric Variables")
    default_index = numeric_columns.index("life_expectancy") if "life_expectancy" in numeric_columns else 0
    dist_var = st.selectbox(
        "Select a numeric variable",
        options=numeric_columns,
        index=default_index
    )
    hist_df = filtered_df.dropna(subset=[dist_var])
    hist_fig = px.histogram(
        hist_df,
        x=dist_var,
        color="region_name",
        nbins=30,
        title=f"Distribution of {dist_var.replace('_',' ').title()} by Region"
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    box_df = filtered_df.dropna(subset=[dist_var])
    box_fig = px.box(
        box_df,
        x="region_name",
        y=dist_var,
        color="region_name",
        title=f"{dist_var.replace('_',' ').title()} Boxplot by Region"
    )
    st.plotly_chart(box_fig, use_container_width=True)

# -----------------------------
# Tab 3: Trends Over Time
# -----------------------------
with tab3:
    st.subheader("Trends Over Time")
    default_index = numeric_columns.index("health_expenditure_pct_gdp") if "health_expenditure_pct_gdp" in numeric_columns else 0
    y_var = st.selectbox(
        "Select variable for y-axis",
        options=numeric_columns,
        index=default_index
    )
    years = sorted(filtered_df['year'].dropna().unique())
    year_range = st.slider(
        "Select Year Range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )
    trend_df = filtered_df[
        (filtered_df['year'] >= year_range[0]) &
        (filtered_df['year'] <= year_range[1])
    ].copy()
    
    plot_option = st.radio("Trend View", options=["Average by Region", "By Selected Countries"])
    
    if plot_option == "Average by Region":
        trend_plot_df = trend_df.groupby(["region_name", "year"])[y_var].mean().reset_index()
        trend_fig = px.line(
            trend_plot_df,
            x="year",
            y=y_var,
            color="region_name",
            markers=True,
            title=f"{y_var.replace('_',' ').title()} Over Time (Average by Region)"
        )
    else:
        countries = trend_df['country'].unique()
        selected_countries_trend = st.multiselect(
            "Select Countries",
            options=countries,
            default=countries[:5]
        )
        trend_plot_df = trend_df[trend_df['country'].isin(selected_countries_trend)]
        trend_fig = px.line(
            trend_plot_df,
            x="year",
            y=y_var,
            color="country",
            line_group="country",
            markers=True,
            title=f"{y_var.replace('_',' ').title()} Over Time by Country"
        )
    st.plotly_chart(trend_fig, use_container_width=True)

# -----------------------------
# Tab 4: Relationships
# -----------------------------
with tab4:
    st.subheader("Life Expectancy vs Other Variables")
    x_options = [col for col in numeric_columns if col != "life_expectancy"]
    default_index = x_options.index("gdp_per_capita") if "gdp_per_capita" in x_options else 0
    x_var = st.selectbox(
        "Select variable for x-axis",
        options=x_options,
        index=default_index
    )
    rel_df = filtered_df.dropna(subset=[x_var, 'life_expectancy', 'population'])

    rel_fig = px.scatter(
        rel_df,
        x=x_var,
        y='life_expectancy',
        color='region',
        size='population',  # safe now, no NaNs
        hover_name='country',
        trendline='ols',
        title=f"Life Expectancy vs {x_var.replace('_',' ').title()}")
    
    st.plotly_chart(rel_fig, use_container_width=True)

    if len(rel_df) > 2:
        corr_value = rel_df[x_var].corr(rel_df["life_expectancy"])
        st.markdown(f"**Correlation:** {corr_value:.2f}")

# -----------------------------
# Tab 5: Map
# -----------------------------
with tab5:
    st.subheader("Global Health Indicator Map")
    map_var_options = ["life_expectancy", "health_expenditure_pct_gdp", "gdp_per_capita", "infant_deaths", "adult_mortality"]
    default_index = map_var_options.index("life_expectancy")
    map_var = st.selectbox("Select variable to display", options=map_var_options, index=default_index)

    map_df = filtered_df.dropna(subset=[map_var]).sort_values("year")

    map_fig = px.choropleth(
        map_df,
        locations="country",
        locationmode="country names",
        color=map_var,
        hover_name="country",
        hover_data=["year", "life_expectancy", "health_expenditure_pct_gdp", "gdp_per_capita"],
        color_continuous_scale="Viridis",
        animation_frame="year",
        title=f"{map_var.replace('_',' ').title()} Across Countries Over Time"
    )
    st.plotly_chart(map_fig, use_container_width=True)



