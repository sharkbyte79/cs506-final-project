import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_random_forest(X_train: DataFrame, y_train: DataFrame):
    # initialize the random forest regressor
    model: RandomForestRegressor = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ) 

    model.fit(X_train, y_train)

