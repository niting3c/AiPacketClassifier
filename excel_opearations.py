import pandas as pd


class ExcelOperations:
    def read_xlsx(self, file='base_knowledge.xlsx', sheet_name='base'):
        df = pd.read_excel(file, sheet_name=sheet_name)
        result = {}

        for col_name in df.columns[1:]:
            boolean_values = df[col_name].astype(bool).tolist()
            result[col_name] = boolean_values
        return result

    def create_excel_file(self, data_list, file_name):
        # Create a Pandas Excel writer with the file_name
        writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

        for data in data_list:
            model_name = data['model_name']
            sheet_data = {}

            for key, value in data.items():
                if key != 'model_name':
                    sheet_data[key] = value

            df = pd.DataFrame(sheet_data)
            df.to_excel(writer, sheet_name=model_name, index_label="packet_number")

        writer.save()


result = ExcelOperations().read_xlsx()
