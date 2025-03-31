from pandas import DataFrame, read_csv, to_datetime
from sklearn.model_selection import train_test_split


# wrapper functions for data manipulation and preprocessing

def csv_to_df(fpath: str) -> DataFrame:
    """Returns the csv specified by fpath as a DataFrame with basic processing"""
    if not fpath.endswith(".csv"):
        raise TypeError(f"Parameter fpath: {fpath} does not specify a csv")

    df: DataFrame = read_csv(
        fpath, na_values={"Area of Responsibility (AOR)": ["Unmapped AOR Records"]}
    )  # Convert rows where AOR is 'Unmapped' to NaN

    # shorten long column names
    df = df.rename(columns={"Area of Responsibility (AOR)": "AOR", "Country of Citizenship": "Citizenship", "Administrative Arrests": "Arrests"})

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


def split_data(df: DataFrame) -> tuple[DataFrame, ...]:
    """Returns a split of the given data as train and test sets"""
    x_train, x_test, y_train, _ = train_test_split(
        df, random_state=42
    )  # discard y_test

    return x_train, x_test, y_train  # pyright: ignore
