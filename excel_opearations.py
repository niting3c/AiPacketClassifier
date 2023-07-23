import pandas as pd


class ExcelOperations:
    def read_xlsx(self, file='base_knowledge.xlsx', sheet_name='base'):
        df = pd.read_excel(file, sheet_name=sheet_name)
        result = {}

        for col_name in df.columns[1:]:
            boolean_values = df[col_name].astype(bool).tolist()
            result[col_name] = boolean_values
        return result

    def create_excel_file(self, data, file_name="models_output.xlsx"):
        # Create a Pandas Excel writer with the file_name
        writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

        for d in data:
            sheet_name = d["name"]
            for item in d["items"]:
                df = pd.DataFrame(item)
                df.to_excel(writer, sheet_name=sheet_name, index_label="packet_number")
        writer.close()


result = ExcelOperations().read_xlsx()

# Example usage:
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

ExcelOperations().create_excel_file(json_data, "test.xlsx")
