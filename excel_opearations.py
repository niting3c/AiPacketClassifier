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

    def create_excel_file(self, data, file_name="models_output.xlsx"):
        """
        Creates an Excel (.xlsx) file using the provided data.

        Args:
            data (list): A list of dictionaries containing sheet names and corresponding data.
                Example data: [{
                    "name": "sheet_name",
                    "items": [{"column1": [True, False, True], "column2": [False, True, False]}, ...]
                }, ...]
            file_name (str, optional): The name of the output Excel file. Defaults to "models_output.xlsx".
        """
        try:
            # Create a Pandas Excel writer with the file_name
            writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

            for d in data:
                sheet_name = d["name"]
                for item in d["items"]:
                    df = pd.DataFrame(item)
                    df.to_excel(writer, sheet_name=sheet_name, index_label="packet_number")
            writer.close()
            print(f"Excel file '{file_name}' created successfully.")
        except Exception as e:
            print(f"Error creating Excel file: {e}")


if __name__ == "__main__":
    # Example usage:

    # Read data from 'base_knowledge.xlsx' and store it in 'result'
    result = ExcelOperations().read_xlsx()

    # Example data in JSON format
    json_data = [{
        "name": "facebook1",
        "items": [
            {
                "sqlInjection": [True, False, False],
                "drupal_attack": [True, False, True],
            }]
    }, {
        "name": "facebook2",
        "items": [
            {
                "sqlInjection": [True, False, False],
                "drupal_attack": [True, False, True],
            }]
    }]

    # Create an Excel file with the provided data and file name
    ExcelOperations().create_excel_file(json_data, "test.xlsx")
