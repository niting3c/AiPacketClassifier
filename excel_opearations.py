import csv

import pandas as pd


class ExcelOperations:
    def read_xlsx(self, file='base_knowledge.xlsx', sheet_name='base'):
        """
        Reads data from an Excel (.xlsx) file and returns it as a dictionary.

        Args:
            file (str, optional): The path of the Excel file. Defaults to 'base_knowledge.xlsx'.
            sheet_name (str, optional): The name of the sheet to read from. Defaults to 'base'.

        Returns:
            dict: A dictionary containing column names as keys and boolean lists as values.
        """
        try:
            df = pd.read_excel(file, sheet_name=sheet_name)
            base_result = {}

            for col_name in df.columns[1:]:
                boolean_values = df[col_name].astype(bool).tolist()
                base_result[col_name] = boolean_values
            return base_result
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return {}

    def write_csv(self, data, csv_file_path="./data/mixed_data.csv"):
        """
        Writes data to a CSV file.

        Args:
            data (list): The data to write to the CSV file.
        """
        try:
            field_names = ["text", "label"]
            # Write the data to the CSV file
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=field_names)
                # Write the header row
                writer.writeheader()
                # Write the data rows
                writer.writerows(data)
            print(f"Data successfully written to {csv_file_path}.")

        except Exception as e:
            print(f"Error writing CSV file: {e}")
