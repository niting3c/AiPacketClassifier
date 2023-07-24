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
