import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    df = df.copy()

    threshold = df["Administrative Arrests"].quantile(0.75)
    df["High Arrest Level"] = (df["Administrative Arrests"] > threshold).astype(int)

    df.fillna("Missing", inplace=True)

    return df

def model_random_forest_regression(df):
    features = ['Criminality', 'AOR', 'Country of Citizenship', 'Fiscal Year', 'Fiscal Quarter', 'Fiscal Month']
    target = 'Administrative Arrests'

    X = df[features]
    y = df[target]

    cat_features = ['Criminality', 'AOR', 'Country of Citizenship']
    num_features = ['Fiscal Year', 'Fiscal Quarter', 'Fiscal Month']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features)
    ])

    model = Pipeline([
        ('prep', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=0))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("\nRandom Forest Regression")
    print("------------------------")
    print("Mean Squared Error:", round(float(mse), 2))

    return model, preprocessor.get_feature_names_out()

def model_knn_classification(df):
    features = ['Criminality', 'AOR', 'Country of Citizenship', 'Fiscal Year', 'Fiscal Quarter', 'Fiscal Month']
    target = 'High Arrest Level'

    X = df[features]
    y = df[target]

    cat_features = ['Criminality', 'AOR', 'Country of Citizenship']
    num_features = ['Fiscal Year', 'Fiscal Quarter', 'Fiscal Month']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features)
    ])

    model = Pipeline([
        ('prep', preprocessor),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nKNN Classification")
    print("------------------")
    print(classification_report(y_test, y_pred))

def analyze_country_bias(df):
    print("\nCountry-Level Arrests Summary")
    print("-----------------------------")
    country_summary = df.groupby("Country of Citizenship")["Administrative Arrests"].sum().sort_values(ascending=False)
    print(country_summary.head(10))

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=country_summary.head(10).values, y=country_summary.head(10).index, palette="Reds_r")
    plt.title("Top 10 Countries by Total Administrative Arrests")
    plt.xlabel("Total Arrests")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(model, feature_names):
    print("\nFeature Importances (Top 15)")
    print("----------------------------")
    importances = model.named_steps['rf'].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = np.array(feature_names)[sorted_idx][:15]
    top_importances = importances[sorted_idx][:15]

    for name, val in zip(top_features, top_importances):
        print(f"{name}: {round(val, 4)}")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features)
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()

def run_all(df):
    df = preprocess_data(df)

    model, feature_names = model_random_forest_regression(df)
    model_knn_classification(df)
    analyze_country_bias(df)
    analyze_feature_importance(model, feature_names)

