import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from pathlib import Path
import seaborn as sns


from cs506_final_project.process import csv_to_df

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILE = "ICE_data.csv"
FILEPATH = f"{PROJECT_ROOT}/data/raw/{TARGET_FILE}"




def plot_bar_counts(
    df,
    column_name,
    ascending=True,
    top_n=None,
    integer_ticks=False,
    missing_label="Missing",
    try_parse_dates=False,
    horizontal_if_many=True,
    horizontal_threshold=10
):
    series = df[column_name].fillna(missing_label)
    value_counts = series.value_counts()

    # Sorting
    if try_parse_dates:
        date_index = pd.to_datetime(value_counts.index, format="%b %Y", errors="coerce")
        if date_index.notna().all():
            sorted_index = date_index.sort_values(ascending=ascending).index
            value_counts = value_counts.iloc[sorted_index]
        else:
            value_counts = value_counts.sort_values(ascending=ascending)
    else:
        index_numeric = pd.Series(pd.to_numeric(pd.Series(value_counts.index), errors="coerce"))
        if index_numeric.notna().all():
            sorted_index = index_numeric.sort_values(ascending=ascending).index
            value_counts = value_counts.iloc[sorted_index]
        else:
            value_counts = value_counts.sort_values(ascending=ascending)

    if top_n:
        value_counts = value_counts.head(top_n)

    use_horizontal = horizontal_if_many and len(value_counts) > horizontal_threshold
    kind = "barh" if use_horizontal else "bar"

    # Dynamic figure size
    if use_horizontal:
        fig_height = max(0.4 * len(value_counts), 6)
        fig, ax = plt.subplots(figsize=(12, fig_height))
    else:
        fig_width = max(0.5 * len(value_counts), 10)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

    value_counts.plot(kind=kind, ax=ax)

    ax.set_title(f'Count of {column_name}')
    ax.set_ylabel('Number of Records' if not use_horizontal else column_name)
    ax.set_xlabel(column_name if not use_horizontal else 'Number of Records')

    if use_horizontal:
        # Explicitly set all ticks and labels to avoid skipping
        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels(value_counts.index, fontsize=10)
        plt.subplots_adjust(left=0.3)  # More room for long labels
    else:
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=60, ha='right', fontsize=10)
        plt.subplots_adjust(bottom=0.3)

    if integer_ticks:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if use_horizontal:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()




def plot_time_series_by_year(df, year_column):
    df = df.copy()

    df[year_column] = df[year_column].astype(str).str.extract(r'(\d{4})')[0]
    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')

    # Drop NaNs before converting to int
    df = df.dropna(subset=[year_column])
    df[year_column] = df[year_column].astype(int)

    # Group and sort by year
    year_counts = df[year_column].value_counts().sort_index()

    ax = year_counts.plot(marker='o')
    ax.set_title("Detainments per Fiscal Year")
    ax.set_xlabel(year_column)
    ax.set_ylabel("Number of Records")

    # Force integer ticks on both axes
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_time_series(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    ts = df[date_column].value_counts().sort_index()
    ts = ts.resample('M').sum()  # monthly counts
    ts.plot()
    plt.title('Detainments Over Time')
    plt.ylabel('Count')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

def plot_heatmap_alt(df, groupby_col):
    grouped = df[groupby_col].value_counts().unstack(fill_value=0)
    sns.heatmap(grouped, cmap="Blues", linewidths=.5)
    plt.title(f'Geographic Heatmap by {groupby_col}')
    plt.tight_layout()
    plt.show()

def plot_heatmap(df, row_col, col_col):
    cross_tab = pd.crosstab(df[row_col], df[col_col])
    sns.heatmap(cross_tab, cmap="Blues", linewidths=.5)
    plt.title(f"Heatmap: {row_col} vs {col_col}")
    plt.ylabel(row_col)
    plt.xlabel(col_col)
    plt.tight_layout()
    plt.show()

def plot_distribution(df, column_name, integer_ticks=False, bins=7):
    data = df[column_name].dropna()

    ax = data.plot(kind='hist', bins=bins, alpha=0.7)
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.tight_layout()

    if integer_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
    # df[column_name].dropna().plot(kind='box')
    # plt.title(f'Box Plot of {column_name}')
    # plt.tight_layout()
    # plt.show()



def print_column_stats(df, column_name):
    print(f"Stats for {column_name}")
    print(df[column_name].describe())

def print_all_stats(df):
    for col in df:
        print_column_stats(df, col)

def main():
    dataframe = csv_to_df(FILEPATH)
    print("Dataframe: ", dataframe)
    print_all_stats(dataframe) # -- print out summary for each df column

    # bar plots
    for col_name in dataframe:
        shouldParse = False
        if col_name == "Month-Year":
            shouldParse = True

        plot_bar_counts(
            df = dataframe,
            column_name = col_name,
            ascending=True,
            integer_ticks=True,
            try_parse_dates = shouldParse
        )
    plot_time_series_by_year(dataframe, "Fiscal Year")

    plot_heatmap(dataframe, "Fiscal Year", "Country of Citizenship")
    plot_heatmap(dataframe, "Fiscal Year", "AOR")
    plot_heatmap(dataframe, "AOR", "Criminality")
    plot_distribution(dataframe, "Fiscal Year", integer_ticks=True)


if __name__ == "__main__":
    main()
