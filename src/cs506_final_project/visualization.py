from pathlib import Path
from cs506_final_project.process import csv_to_df

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILE = "mini_ICE_data.csv"
FILEPATH = f"{PROJECT_ROOT}/data/raw/{TARGET_FILE}"

dataframe = csv_to_df(FILEPATH)
print(dataframe)

