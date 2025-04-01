---
title: CS506
author: Pardesh Dhakal, Layne Pierce
date: 2-9-2025
---



# CS506 Final Project
This project will focus on **ICE** (U.S Immigration and Customs Enforcement) arrests, detentions, expulsions and removals.

## Use


[https://python-poetry.org/docs/]


### Files

***Check section Make commands -- shortcuts for Poetry commands present***

`pyproject.toml`: This file stores *metadata* (author name, version, description, etc.). Includes a dependency declaration and a configuration for build tools. Meant to be edited by developers.

`poetry.lock`: This file stores the *exact dependency* information and guarantee a deterministic, reproducible environment for colloborators. It is automatically generated by poetry.

### Makefile

You *could* run the poetry commands (documented in the following section) by themselves but `Layne` added Makefile commands to make it easier .

***TLDR:*** run `make help`

```bash
make install # install dependencies for the project

make clean # clean environment

make activate # activate the development environment

make deactivate # deactivate

make help # to see all available targets

```

### Poetry commands


```bash
# make sure poetry is installed (check documentation above)

poetry install # will install dependencies specified by pyproject and poetry.lock

poetry add <package-name> # will add <package-name> to list of dependencies in poetry.lock :)

poetry remove <package-name> # remove a dependency

poetry shell # how to activate and work within the developer environment

# There are options to create sub groups within environments but that is overly complicated for our purposes.

```

The general order of commands is as follows:

`make Poetry` (if not installed)

`make install` (get dependencies)

`make activate` (run environment)

`poetry [add or remove]` (to add necessary packages) 

Poetry version: -------------------------------------------------------------------------------------------------------

`poetry install` (after freshly cloning the repo)

`poetry shell` (to develop within the development environment)

`poetry [add or remove] package-name` (as needed over the course of development)

## Goal(s)
From a general browsing of the yearly trends, we can see a consistent (and disconcerting) increase in all forms of activity throughout the country. The goal of this project is to identify and measure the extent to which specific demographics are disproportionately affected by anti-migrant operations, as well as nalyze regional trends such as locations where ICE activity is higher and making informed predictions on future trends.

With the current discourse and political situation regarding this topic in the United States, we believe it is important to have a strong understanding of the government’s actions. With that being said, it is possible that the goals of the project will change through working with the data.

## Data
Presently, Official [ICE enforcement and removal statistics](https://www.ice.gov/spotlight/statistics)


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

## Results



