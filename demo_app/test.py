import pandas as pd

input_df = pd.read_csv('input.csv')
template_df = pd.read_csv('template.csv')

train_features = set(template_df.columns)
new_features = set(input_df.columns)

print("Missing features:", train_features - new_features)
print("Extra features:", new_features - train_features)