from pathlib import Path
from cs506_final_project.process import csv_to_df

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILE = "mini_ICE_data.csv"
FILEPATH = f"{PROJECT_ROOT}/data/raw/{TARGET_FILE}"


def plot_bar_count(df, column_name):
    pass

def plot_time_series(df, date_column):
    pass

def plot_heatmap(df, groupby_col):
    pass

def plot_distribution(df, column_name):
    pass

def print_column_stats(df, column_name):
    print(f"Stats for {column_name}")
    print(df[column_name].describe())

def print_all_stats(df):
    for col in df:
        print_column_stats(df, col)

def main():
    dataframe = csv_to_df(FILEPATH)
    print("Dataframe: ", dataframe)
    print_all_stats(dataframe)

if __name__ == "__main__":
    main()
