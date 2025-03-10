import pandas as pd
import os

class FileData:
    def __init__(self, file_path: str, encoding: str = 'utf-8', delimiter = '\t') -> None:
        """
        Initializes FileData object with a given file path.
        :param file_path: The string path to a file
        :param encoding: The encoding format for reading the file (default is 'utf-8')
        """
        self.file_path = file_path
        self.encoding = encoding
        self.data_frame = None
        self.delimiter = delimiter

    def create_df(self) -> None:
        """
        Reads a CSV file and turns it into a DataFrame.
        """
        try:
            # Directly read the CSV file into the DataFrame with specified encoding
            self.data_frame = pd.read_csv(self.file_path, encoding=self.encoding, delimiter = self.delimiter)
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_file_info(self) -> None:
        if self.data_frame is not None:
            print("\nHead: ", self.data_frame.head())
            print("\n\nInfo: ", self.data_frame.info())
            print("\n\nDescribe: ", self.data_frame.describe())

    def get_df(self):
        if self.data_frame is not None:
            return self.data_frame
        else:
            print("Data frame is empty")


def main():
    file_path = '../data/raw/BarChart_Arrests.csv'  # Adjust path if needed
    encoding = 'utf-16'  # Change encoding as needed
    bar_arrest_data = FileData(file_path, encoding)
    bar_arrest_data.create_df()
    bar_arrest_data.get_file_info()
    bar_arrest_df = bar_arrest_data.get_df()


if __name__ == '__main__':
    main()

