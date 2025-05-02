import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path
import plotly.express as px
import os

from process import (
    csv_to_df,
    arrests_in_aor_by_month_year,
    arrests_by_citizenship,
)
from visualization import (
    arrest_density_bubble_mapped,
    citizenship_bubble_mapped,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILE = "ICE_data.csv"
FILEPATH = f"{PROJECT_ROOT}/data/raw/{TARGET_FILE}"

CSV_PATH = FILEPATH
N_LAGS = 3
RF_ESTIMATORS = 100
RF_RANDOM_STATE = 42
HORIZON = 6
FREQ = "ME"

# loading our dataframe
df = pd.read_csv(CSV_PATH)
df = df.rename(
    columns={
        "Month-Year": "ds",
        "Administrative Arrests": "y",
        "Country of Citizenship": "Citizenship",
    }
)
df["ds"] = pd.to_datetime(df["ds"], format="%b %Y")

# Here we build our monthly panel per Citizenship., this is to potentially notice trends that will indicate discrimination
eths = df["Citizenship"].unique()
idx = pd.date_range(df["ds"].min(), df["ds"].max(), freq=FREQ)
panels = []
for eth in eths:
    sub = df[df.Citizenship == eth].set_index("ds")
    monthly = sub["y"].resample(FREQ).sum().reindex(idx, fill_value=0)
    tmp = monthly.to_frame().rename_axis("ds").reset_index()
    tmp["Citizenship"] = eth
    panels.append(tmp)
panel = pd.concat(panels, ignore_index=True)


# feature engineering
def make_features(data, n_lags=N_LAGS):
    g = data.sort_values("ds").copy()
    g["t"] = np.arange(len(g))
    g["month_sin"] = np.sin(2 * np.pi * g["ds"].dt.month / 12)
    g["month_cos"] = np.cos(2 * np.pi * g["ds"].dt.month / 12)
    for lag in range(1, n_lags + 1):
        g[f"y_lag{lag}"] = g["y"].shift(lag)
    g["roll_mean_3"] = g["y"].shift(1).rolling(3).mean()
    g["roll_std_3"] = g["y"].shift(1).rolling(3).std().fillna(0)
    return g.dropna()


feature_cols = (
    ["t", "month_sin", "month_cos"]
    + [f"y_lag{i}" for i in range(1, N_LAGS + 1)]
    + ["roll_mean_3", "roll_std_3"]
)

# Split into training & validation by time
last_date = panel["ds"].max()
offset = pd.tseries.frequencies.to_offset(FREQ)
cutoff = last_date - HORIZON * offset

train_panel = panel[panel["ds"] <= cutoff]
val_panel = panel[(panel["ds"] > cutoff) & (panel["ds"] <= last_date)]

# Prepare training set
train_frames = []
for eth, grp in train_panel.groupby("Citizenship"):
    train_frames.append(make_features(grp))
train_df = pd.concat(train_frames, ignore_index=True)
X_train = train_df[feature_cols]
y_train = train_df["y"]

# Train model
model = RandomForestRegressor(n_estimators=RF_ESTIMATORS, random_state=RF_RANDOM_STATE)
model.fit(X_train, y_train)

# Validate on held-out window, compute Euclidean distances
val_preds = []
for eth, grp in train_panel.groupby("Citizenship"):
    buf = grp[["ds", "y"]].copy().reset_index(drop=True)
    future_dates = pd.date_range(start=cutoff + offset, periods=HORIZON, freq=FREQ)
    for dt in future_dates:
        feats = make_features(buf).iloc[[-1]][feature_cols]
        y_pred = model.predict(feats.values)[0]
        actual = val_panel[(val_panel.Citizenship == eth) & (val_panel.ds == dt)]["y"]
        y_true = actual.iloc[0] if not actual.empty else np.nan
        val_preds.append(
            {
                "Citizenship": eth,
                "ds": dt,
                "y_true": y_true,
                "y_pred": y_pred,
                "error_euclidean": np.sqrt((y_true - y_pred) ** 2),
            }
        )
        buf = pd.concat(
            [buf, pd.DataFrame([{"ds": dt, "y": y_true}])], ignore_index=True
        )

val_df = pd.DataFrame(val_preds).dropna(subset=["y_true"])
# overall MSE
mse = mean_squared_error(val_df["y_true"], val_df["y_pred"])
print(f"Validation MSE over last {HORIZON} months: {mse:.2f}")

# Euclidean distance per Citizenship (across the horizon)
group_euclid = (
    val_df.groupby("Citizenship")
    .apply(lambda g: np.sqrt(np.sum((g.y_true - g.y_pred) ** 2)))
    .reset_index(name="euclidean_distance")
)
print("\nEuclidean distance per Citizenship over validation window:")
print(group_euclid)

print("\nPer-prediction Euclidean errors:")
print(val_df[["Citizenship", "ds", "y_true", "y_pred", "error_euclidean"]])
FUTURE_HORIZON = 12

# build rolling forecasts exactly as in validation, but now for 12 months beyond your last date
future_preds = []
for eth in eths:
    # start with full history for this Citizenship
    buf = panel[panel.Citizenship == eth][["ds", "y"]].copy().reset_index(drop=True)
    future_dates = pd.date_range(start=panel["ds"].max() + offset, periods=FUTURE_HORIZON, freq=FREQ)
    for dt in future_dates:
        feats = make_features(buf).iloc[[-1]][feature_cols]
        y_fore = model.predict(feats.values)[0]
        future_preds.append({"Citizenship": eth, "ds": dt, "y_pred": y_fore})
        # append the forecast back into buf so next lag/roll works
        buf = pd.concat([buf, pd.DataFrame([{"ds": dt, "y": y_fore}])], ignore_index=True)

future_df = pd.DataFrame(future_preds)


# 1) Line chart: Forecast Next 12 Months for Top 5 Citizenships by Recent Volume
# pick top5 by sum of last 6 months of actuals
recent = panel[panel.ds > panel.ds.max() - 6 * offset]
top5 = recent.groupby("Citizenship")["y"].sum().nlargest(5).index.tolist()
df_top5_future = future_df[future_df["Citizenship"].isin(top5)]

fig1 = px.line(
    df_top5_future,
    x="ds",
    y="y_pred",
    color="Citizenship",
    title="12-Month Forecast of Arrests (Top 5 Citizenship)",
    labels={"ds": "Date", "y_pred": "Predicted Arrests"}
)
fig1.update_layout(xaxis_tickangle=-45)
fig1.show()


# 2) Faceted small-multiples: 12-Month Forecast for All Citizenships
fig2 = px.line(
    future_df,
    x="ds",
    y="y_pred",
    facet_col="Citizenship",
    facet_col_wrap=4,
    height=1400,
    title="12-Month Forecast for Each Citizenship"
)
# clean up facet labels (remove "Citizenship=" prefix)
for anno in fig2.layout.annotations:
    anno.text = anno.text.split("=")[-1]
fig2.update_yaxes(matches=None)
fig2.update_xaxes(tickangle=-45)
fig2.update_layout(showlegend=False)
fig2.show()