from pandas import DataFrame
import numpy as np
import os
from cs506_final_project.process import csv_to_df

def test_csv_to_df():
    want: DataFrame = DataFrame(
        {
            "Criminality": [
                "Pending Criminal Charges",
                "Criminal Conviction",
                "Criminal Conviction",
                "Criminal Conviction",
                "Criminal Conviction",
            ],
            "Area of Responsibility (AOR)": [np.nan, "Atlanta", "Atlanta", "Atlanta", "Atlanta"],
            "Country of Citizenship": [
                "MEXICO",
                "COLOMBIA",
                "COLOMBIA",
                "COLOMBIA",
                "COLOMBIA",
            ],
            "Fiscal Year": [2021, 2023, 2024, 2024, 2024],
            "Fiscal Quarter": [1, 2, 1, 4, 4],
            "Fiscal Month": [1, 5, 1, 10, 11],
            "Month-Year": ["10-2020", "02-2023", "10-2023", "07-2024", "08-2024"],
            "Administrative Arrests": [10, 18, 16, 11, 11],
        }
    )


    fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/mini_ICE_data.csv"))

    assert csv_to_df(fpath).equals(want)