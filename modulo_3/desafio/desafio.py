import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('indian_liver_patient.csv',
                 names=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "Class"])
print(df.head(10))
print(df.describe())
print(df.info())
print(df.shape)
print(df.isnull().sum())