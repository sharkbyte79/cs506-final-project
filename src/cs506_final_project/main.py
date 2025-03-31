import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from pathlib import Path
import seaborn as sns

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from cs506_final_project.ice_modeling import run_all


from cs506_final_project.process import csv_to_df

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILE = "ICE_data.csv"
# OTHER_TARGET = "Over_Time_Years_Removals.csv"
FILEPATH = f"{PROJECT_ROOT}/data/raw/{TARGET_FILE}"

def main():
    dataframe = csv_to_df(FILEPATH)
    print(dataframe)
    run_all(dataframe)



if __name__ == "__main__":
    main()
