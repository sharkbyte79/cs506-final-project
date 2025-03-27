from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from datetime import date, datetime


class DataPreprocessor:
    """
    A class containing wrapper methods for data manipulation and preprocessing
    """

    def csv_to_df(fpath: str) -> DataFrame:
        """Returns the csv specified by fpath as a DataFrame with basic processing"""
        if not fpath.endswith(".csv"):
            raise TypeError(f"Parameter fpath: \'{fpath}\' does not specify a csv")

        df: DataFrame = read_csv(
            fpath, na_values={"Area of Responsibility (AOR)": ["Unmapped"]}
        )  # Convert rows where AOR is 'Unmapped' to NaN

        # TODO: Standardize format for features including date/month/year

        # NOTE: Possibly sort other date-time fields as well
        df.sort_values(
            by=["Month-Year"], ascending=False, inplace=True
        )  # Sort in descending order by 'Month-Year'

        return df

    def split_data(df: DataFrame) -> tuple[DataFrame, ...]:
        """Returns a split of the given data as train and test sets"""
        x_train, x_test, y_train, _ = train_test_split(
            df, random_state=42
        )  # discard y_test
        ...