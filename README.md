
---
title: CS506
author: Pardesh Dhakal, Layne Pierce
date: 3-31-2025
---

# CS506 Final Project
This project will focus on **ICE** (U.S Immigration and Customs Enforcement) arrests, detentions, expulsions and removals.

See the midterm presentation [here](https://youtu.be/zlY1djLtqp8).

## Setup

To run the source (`src`) project files, start with creating a virtual environment. 

`python3 -m venv env`

Now you need to activate the virtual environment: `source path/to/env/bin/activate`.

From there, you can install the dependencies. 

`pip3 install -r /path/to/requirements.txt`


To add dependencies:


`pip3 install [package-name]` and then add it to the dependency list `pip3 freeze > requirements.txt`

From there, run the desired files:

`python3 [file-name].py`


### Include links to download data below

## Goal(s)
From a general browsing of the yearly trends, we can see a consistent (and disconcerting) increase in all forms of activity throughout the country. The goal of this project is to identify and measure the extent to which specific demographics are disproportionately affected by anti-migrant operations, as well as nalyze regional trends such as locations where ICE activity is higher and making informed predictions on future trends.

With the current discourse and political situation regarding this topic in the United States, we believe it is important to have a strong understanding of the government’s actions. With that being said, it is possible that the goals of the project will change through working with the data.

## Data
Presently, Official [ICE enforcement and removal statistics](https://www.ice.gov/spotlight/statistics)
We have specifically used the 2021-2024 detainment data.


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

## Results

### Visualization results

The most glaring result from the visualizations was the significant over-representation of arrests against undocumented people of prior Mexican nationality. In general, we were able to notice the over-representation of Latin-American countries. In regards to arrest frequency based on area of residence, we found that the cities New York, Houston, Miami, and Los Angeles suggest a high-enforcement intensity. Another significant discovery was the majority of those arrested were due to the reason of *Other Immigration violator* or in other words, the singular crime of being undocumented.


### Modeling Results

Results from random forest regression showed us that features such as `Fiscal Year` and `Fiscal Month` had the highest importance. As mentioned from what we noticed in the visualizations, this modeling confirmed the significance of the `Area of Residence (AOR)` and prior Mexican citizenship as features that strongly influence arrest outcomes, suggesting a bias in enforcement patterns tied to both geography and nationality. The finding that the majority of these arrests were based on soley immigration status violations add to the significance of this overt over-representation.


The results from our KNN modeling:
- Accuracy: 86%
- Precision and Recall (High arrest class): ~70%

This tells us that high arrests are predictable, based off location and nationality, large-scale detainments can be anticipated.
