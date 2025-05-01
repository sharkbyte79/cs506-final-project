from pandas import DataFrame, read_csv, to_datetime, concat, Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time
from geopy.geocoders import Nominatim


# wrapper functions for data manipulation and preprocessing


def csv_to_df(fpath: str) -> DataFrame:
    """Returns the csv specified by fpath as a DataFrame with basic processing"""
    if not fpath.endswith(".csv"):
        raise TypeError(f"Parameter fpath: {fpath} does not specify a csv")

    df: DataFrame = read_csv(
        fpath, na_values={"Area of Responsibility (AOR)": ["Unmapped AOR Records"]}
    )  # Convert rows where AOR is 'Unmapped' to NaN

    # shorten long column names
    df = df.rename(
        columns={
            "Area of Responsibility (AOR)": "AOR",
            "Country of Citizenship": "Citizenship",
            "Administrative Arrests": "Arrests",
        }
    )

    # Convert 'Month-Year features to datetime objects for sorting
    df["Month-Year"] = to_datetime(df["Month-Year"], format="%b %Y")

    # NOTE: Possibly sort other date-time fields as well
    df = df.sort_values(
        by="Month-Year", ascending=True
    )  # ascending/descending is backwards (relative to the view of a DF when printed)

    # to_datetime adds the first day of the month to each Month-Year field by default
    # Convert them back to formatted strings to strip this
    df["Month-Year"] = df["Month-Year"].dt.strftime("%b %Y")

    df["Id"] = df.index # ease of reference

    return df


def aor_to_longitude(df: DataFrame) -> DataFrame:
    """Returns a version of the given DataFrame df with AOR converted to latitude and longitude features"""
    # NOTE: This function is SLOW in order to respect OpenStreetMap's rate limit
    df = df.dropna(subset=["AOR"])
    geolocator = Nominatim(
        user_agent="pierce77@bu.edu"
    )  # must pass email to avoid being blocked by openstreetmap api
    latlong_cache: dict[str, Series] = {}
    df[["Latitude", "Longitude"]] = df["AOR"].apply(
        lambda aor: get_latlong(aor, latlong_cache, geolocator)
    )
    # return df.drop("AOR", axis=1)
    return df


def get_latlong(aor: str, cache: dict[str, Series], geolocator: Nominatim) -> Series:
    """Gets the latitude and longitude for an AOR"""
    # just return the latlong if we've already received it
    if aor in cache:
        return cache[aor]

    time.sleep(1)  # sleep for a second to respect rate limits
    loc = geolocator.geocode(aor.lower())
    cache[aor] = Series([loc.latitude, loc.longitude])  # store this latlong for later
    return cache[aor]


def arrests_in_aor_by_month_year(df: DataFrame) -> DataFrame:
    "Returns a DataFrame containing entries for the number of arrests per each Month-Year in an AOR"
    # Create a new DataFrame where there is an entry for each AOR's total number of arrests for a given Month-Year
    grouped_df = df.groupby(["AOR", "Latitude", "Longitude", "Month-Year"]).agg(
        {"Arrests": "sum"}
    )

    grouped_df = grouped_df.reset_index()
    grouped_df["Month-Year"] = to_datetime(grouped_df["Month-Year"], format="%b %Y")
    grouped_df = grouped_df.sort_values("Month-Year")
    grouped_df["Month-Year"] = grouped_df["Month-Year"].dt.strftime("%b %Y")


    return grouped_df


def encode_and_scale(
    train_df: DataFrame, test_df: DataFrame, id: str, target: str, categorial: list[str]
):
    """Returns a version of the given DataFrame df with label encoding and standard scaling applied"""

    numerical: list[str] = [
        col for col in train_df.columns if col not in [id, target, *categorial]
    ]
    for col in categorial:
        encoder: LabelEncoder = LabelEncoder()  # Create encoder for each iteration
        combined: DataFrame = concat(
            [train_df[col], train_df[col]]
        )  # Combine the columns for consistent label encoding
        encoder.fit(combined)

        train_df[col] = encoder.transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])

    # Using standard scaling on each numerical column
    # Every value is considered equally important when scaled with mean + stdev
    scaler: StandardScaler = StandardScaler()
    train_df[numerical] = scaler.fit_transform(train_df[numerical])
    test_df[numerical] = scaler.fit(test_df[numerical])

    # TODO: define X_train, Y_train, etc for model training


def split_data(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Returns a split of the given data as train and test sets"""
    train, test = train_test_split(df, random_state=42, test_size=0.3)

    map((DataFrame), train, test)

    return train, test
