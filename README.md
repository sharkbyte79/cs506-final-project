---
title: CS506
author: Pardesh Dhakal, Layne Pierce
date: 05-01-2025
---

# CS506 Final Project
This project will focus on **ICE** (U.S Immigration and Customs Enforcement) arrests, detentions, expulsions and removals.

See the midterm presentation [here](https://youtu.be/zlY1djLtqp8).

See the final presentation [here](). 

## IMPORTANT NOTE
For some of the preceding sections, look for a "Final" subsection so that we can demonstrate how our methodology changed or evolved since the midterm report. 

### Files

***Check section Makefile to see how to reproduce our results***

### Makefile

Information will be added here once we figure this stuff out. 

## Goal(s)
From a general browsing of the yearly trends, we can see a consistent (and disconcerting) increase in all forms of activity throughout the country. The goal of this project is to identify and measure the extent to which specific demographics are disproportionately affected by anti-migrant operations, as well as analyze regional trends such as locations where ICE activity is higher and making informed predictions on future trends.

With the current discourse and political situation regarding this topic in the United States, we believe it is important to have a strong understanding of the government’s actions. With that being said, it is possible that the goals of the project will change through working with the data.

### Final Clarified Goal(s)
Through working on this project, this is how we have reworded and revised our goals:

1. Quantify demographic disparities in arrests (by country of citizenship)
2. Build (a) predictive model(s) to forecast future ICE activity

## Data
Presently, Official [ICE enforcement and removal statistics](https://www.ice.gov/spotlight/statistics)
Scope: Monthly detainment data, 2021-2024


## Preprocessing
The processing steps we have taken with the data on arrests by ICE officials begins with a simple conversion from `csv` format to an easily-manipulatable `DataFrame` by the `csv_to_df` function. We also prepare the data for viewing and further processing by sorting chronologically on the "Month-Year" feature, replacing "Unmapped" areas of responsibility with `NaN`, and shortening a number of the longer feature names so that they may be more easily referenced.

Further preprocessing that is more specific to our chosen methods of modeling is performed by establishing an upper threshold for what value for the "Administrative Arrests" feature constitutes a "High Arrest Level" and appending this new feature to the dataframe. This feature is a *binary* value, an arrestee is either "High Arrest Level" or not.


## Modeling
At this time, we have chosen to explore the following modeling methods:
- K-Nearest Neighbors (as discussed in class)
- Random Forests

These methods for classification are ideal for our expressed goal of analyzing *demographic biases* and other discrepancies that may be evident in the **ICE** dataset.

For the **Random Decision Forests**, we perform further preprocessing by normalizing non-target numerical features with **Standard Scaling** (Fiscal quarter, month and year) and [one-hot encoding](https://www.geeksforgeeks.org/ml-one-hot-encoding/) for assigning binary indicators to categorical features (Criminality, area of responsibility and an individual's country of citizenship). The actual modeling consists of an aggregation of 100 decision trees for predictions regarding the number of administrative arrests.

We perform the same preprocessing steps for the  **K-Nearest Neighbors** model which is trained for the same purpose.

### Final Modeling

To forecast future ICE arrest counts by citizenship, we first constructed a continuous monthly panel for each country of citizenship. Any months without recorded arrests were filled with zeroes to ensure that our time series remained uninterrupted.

Next, we engineered a set of features designed for the Random Forest to capture long-term trends and seasonal fluctuations:

1. Time index (`t`): a simple integer counter that increases by one each month.

2. Seasonal encoding: sine and cosine transforms of the month-of-year (e.g. sin(2π·month/12) and cos(2π·month/12)) to represent yearly cyclical patterns.

3. Lagged arrest counts: values from one to three months prior (y_lag1, y_lag2, y_lag3) to help the model learn momentum and short-term autocorrelation.

4. Rolling statistics: a three-month rolling mean and standard deviation, both computed on the previous months, to quantify recent volatility.

We then split our data temporally, holding out the last six months as a validation window. Using the earlier months for training, we performed walk-forward validation: for each citizenship, the model predicted one month ahead, then “observed” the true value and incorporated it into the feature history before predicting the next month.

On this six-month hold-out, the Random Forest achieved:

* Validation MSE: 6 711.42

* Model RMSE: √6 711.42 ≈ 81.9 arrests/month

* Mean true arrests/month: 156.7 (± 580.7)

To ensure real predictive power, we compared against a naïve “persistence” baseline—predicting that each month’s arrest count would equal the previous month’s. That simple approach yielded an RMSE of 6 629.0, meaning our model delivers a 98.8 % reduction in error (1 – 81.9/6 629.0).

Finally, we extended our walk-forward process to generate 12 months of future forecasts for every citizenship. We visualized these as:

1. A line chart showing the twelve-month projection for the top five citizenships, and

2. A small-multiples grid displaying all citizenship forecasts side-by-side.


## Visualization

To better understand patterns in the ICE dataset, we built a set of custom visualizations.

We use bar plots (`plot_bar_counts`) to show how often different values appear in each column. For date-based features like "Month-Year", we sort the bars chronologically. If there are too many categories, the plot switches to a horizontal layout for better readability. Plot sizes also adjust based on how much data is shown.

To explore trends over time, we include two types of time series plots:
- `plot_time_series_by_year`: groups records by fiscal year.
- `plot_time_series`: shows monthly trends using resampled line plots.

We also use heatmaps (`plot_heatmap` and `plot_heatmap_alt`) to visualize how two variables relate to each other.

For numeric columns, we include distribution plots (`plot_distribution`) to show how values are spread out. We add integer tick marks for graphings where decimal numbers don't make sense.

These visual tools are helpful for spotting patterns, trends or biases.

Please See [following directory](data/visualizations/) to view the visualizations we've prepared.

### Final Visualization

In the accompanying `forecast_analysis.ipynb`, we built three key interactive Plotly charts to explore both model validation and future projections:

1. **Validation Error Bar Chart**  
   - After computing per-ethnicity Euclidean error over the 6-month hold-out, we plotted the top 20 citizenships by total error.  
   - This bar chart clearly highlights which groups the model struggles with most—and therefore where we might need more data or specialized models.

2. **Forecast of Top-5 Citizenship Trends**  
   - We identified the five countries with the highest aggregate arrests in the most recent six months.  
   - A multi-series line plot then shows each of these top-5’ twelve-month forecasts side by side, with solid lines for model predictions.  
   - This makes it easy to compare projected trajectories for the groups with the largest recent ICE activity.

3. **Small-Multiples Grid of All Citizenship Forecasts**  
   - For comprehensive coverage, we generated a 4×n grid—one panel per country—each showing its own 12-month forecast.  
   - We reduced tick density and font size to keep the grid legible.  
   - This overview reveals the heterogeneity of future trends: some nationalities are forecast to see rising arrests, others remain near zero, and seasonal cycles emerge even in smaller-volume groups.

Each figure is fully interactive—hover to see exact values, zoom in on time windows, and toggle series visibility—making `forecast_analysis.ipynb` a rich environment for both exploration and presentation.


## Results

### Visualization results

The most glaring result from the visualizations was the significant over-representation of arrests against undocumented people of prior Mexican nationality. In general, we were able to notice the over-representation of Latin-American countries. In regards to arrest frequency based on area of residence, we found that the cities New York, Houston, Miami, and Los Angeles suggest a high-enforcement intensity. Another significant discovery was the majority of those arrested were due to the reason of *Other Immigration violator* or in other words, the singular crime of being undocumented.


### Modeling Results

Results from random forest regression showed us that features such as `Fiscal Year` and `Fiscal Month` had the highest importance. As mentioned from what we noticed in the visualizations, this modeling confirmed the significance of the `Area of Residence (AOR)` and prior Mexican citizenship as features that strongly influence arrest outcomes, suggesting a bias in enforcement patterns tied to both geography and nationality. The finding that the majority of these arrests were based on soley immigration status violations add to the significance of this overt over-representation.


The results from our KNN modeling:
- Accuracy: 86%
- Precision and Recall (High arrest class): ~70%

This tells us that high arrests are predictable, based off location and nationality, large-scale detainments can be anticipated.

### Final Results

#### Model Validation

- **Overall fit**: our Random Forest achieved an RMSE of **81.9 arrests/month** on the 6-month hold-out, compared to a naïve “last-value” RMSE of **6 629**, yielding a **98.8 % reduction in forecasting error**.  
- **Error distribution**: the validation bar chart shows that while most citizenships have low cumulative error, a handful of groups with highly volatile historical counts still drive a disproportionate share of the residuals.  

#### 12-Month Forecast Insights

- **High-volume nationalities** (e.g. Mexico, El Salvador, Honduras) are projected to maintain elevated arrest counts, with gently undulating seasonal cycles.  
- **Mid-range groups** (e.g. Guatemala, Dominican Republic) show flatter trajectories, indicating stable monthly volumes.  
- **Low-volume nationalities** remain near zero but still exhibit faint seasonality—demonstrating the model’s ability to learn patterns even on sparse time series.  

Taken together, these results confirm that our feature pipeline and Random Forest can both accurately reproduce recent trends and generate plausible future scenarios—an essential capability for data-driven policy analysis and resource planning.  









