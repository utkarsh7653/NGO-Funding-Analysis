# NGO Funding Analysis using K Means Clustering and PCA 
Overview
This project analyzes socioeconomic indicators of countries using data from country-data.csv. The goal is to explore and visualize key metrics such as child mortality, income, GDP per capita, and life expectancy to identify patterns and clusters that could inform decision-making, such as prioritizing regions for NGO support. The code includes data cleaning, statistical analysis, and a choropleth map visualization to highlight clusters of countries based on their socioeconomic needs.

Features
Data Cleaning: Handles missing values using pandas to ensure a robust dataset.
Exploratory Data Analysis (EDA): Provides statistical summaries (e.g., mean, median, standard deviation) of variables like child mortality, exports, and health spending.
Visualization: Generates a choropleth map with clusters, where red indicates countries potentially requiring more attention (Cluster 1) and blue represents others (Cluster 0).
Tools Used: Python libraries including pandas, matplotlib, numpy, seaborn, and geopandas (implied for choropleth mapping).

Dataset
The dataset (country-data.csv) contains 167 rows and 10 columns:

country: Name of the country
child_mort: Child mortality rate
exports: Exports as a percentage of GDP
health: Health spending as a percentage of GDP
imports: Imports as a percentage of GDP
income: Per capita income
inflation: Inflation rate
life_expec: Life expectancy
total_fer: Total fertility rate
gdpp: GDP per capita

Results
Statistical Insights: The df.describe() output provides a snapshot of data distribution (e.g., average child mortality = 38.27, average GDP per capita = $12,964).
Choropleth Map: Countries are clustered into two groups:
Cluster 0 (Blue): Generally better socioeconomic conditions.
Cluster 1 (Red): Higher priority areas needing attention (based on clustering criteria not shown in the provided code snippet).
