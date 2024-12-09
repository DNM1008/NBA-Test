import pickle
import pandas as pd
import numpy as np

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
template = pd.read_csv('template.csv')
template = template.drop(columns = ['Unnamed: 0', 'Unnamed: 1'])

# print(template.head())
def match_columns(old_df, new_df):
    # Get the columns that are missing in the new dataset
    missing_columns = [col for col in old_df.columns if col not in new_df.columns]
    
    # Add missing columns with default value 0 to the new dataset
    for col in missing_columns:
        new_df[col] = 0
    
    # Reorder columns to match the order of the old dataset
    all_columns = list(old_df.columns)
    for col in new_df.columns:
        if col not in all_columns:
            all_columns.append(col)
    
    new_df = new_df[all_columns]  # Reorder columns
    return new_df

new_df = pd.read_csv('input.csv')
input_df = match_columns(new_df, template)
# pred = model.predict(input_df)[0]

print(len(new_df.columns))
print(len(template.columns))
# for col in template.columns:
#     print(col)
# print(pred)

a = np.setdiff1d(new_df.columns, template.columns)
print (a)
['D']

print('Done')

