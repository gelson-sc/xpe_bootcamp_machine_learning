import pandas as pd

df = pd.read_csv('indian_liver_patient.csv',
                 names=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "Class"])
print(df.head())
