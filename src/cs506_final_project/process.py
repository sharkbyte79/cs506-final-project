from pandas import DataFrame, read_csv, to_datetime, concat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# wrapper functions for data manipulation and preprocessing

def csv_to_df(fpath: str) -> DataFrame:
    """Returns the csv specified by fpath as a DataFrame with basic processing"""
    if not fpath.endswith(".csv"):
        raise TypeError(f"Parameter fpath: {fpath} does not specify a csv")

    df: DataFrame = read_csv(
        fpath, na_values={"Area of Responsibility (AOR)": ["Unmapped AOR Records"]}
    )  # Convert rows where AOR is 'Unmapped' to NaN

    # shorten name for 'Area of Responsibility (AOR) column
    df = df.rename(columns={"Area of Responsibility (AOR)": "AOR"})

    # Convert 'Month-Year features to datetime objects for sorting
    df["Month-Year"] = to_datetime(df["Month-Year"], format="%b %Y")

    # NOTE: Possibly sort other date-time fields as well
    df = df.sort_values(
        by="Month-Year", ascending=True
    )  # ascending/descending is backwards (relative to the view of a DF when printed)

    # to_datetime adds the first day of the month to each Month-Year field by default
    # Convert them back to formatted strings to strip this
    df["Month-Year"] = df["Month-Year"].dt.strftime("%b %Y")

    return df

def encode_and_scale(train_df: DataFrame, test_df: DataFrame, id: str, target: str, categorial: list[str]):
    """Returns a version of the given DataFrame df with label encoding and standard scaling applied"""

    numerical: list[str] = [col for col in train_df.columns if col not in [id, target, *categorial]]
    for col in categorial:
        encoder: LabelEncoder = LabelEncoder() # Create encoder for each iteration
        combined: DataFrame = concat([train_df[col], train_df[col]]) # Combine the columns for consistent label encoding
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
    train, test = train_test_split(
        df, random_state=42, test_size=0.3
    )  

    map((DataFrame), train, test)

    return train, test

