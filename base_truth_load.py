import pandas as pd
def read_xlsx():
    df= pd.read_excel('base_truth.xlsx', sheet_name='base')
    result={}

    for col_name in df.columns[1:]:
        boolean_values = df[col_name].astype(bool).tolist()
        result[col_name]=boolean_values
    return result
