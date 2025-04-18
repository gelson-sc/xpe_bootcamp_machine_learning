import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

IOT_temp = pd.read_csv("IOT-temp.csv")
# print(IOT_temp.head())
# print(IOT_temp.info())
# print(IOT_temp.isna().sum())
# print(IOT_temp.isnull().sum())
# print(IOT_temp.describe())
print(IOT_temp.groupby("out/in").count()['id'])
# separando o dataset entre valores de In e Out
df_in = IOT_temp[IOT_temp["out/in"] == "In"]
df_out = IOT_temp[IOT_temp["out/in"] == "Out"]
# print(df_in.head(5))
# print(df_out.head(5))
# print(df_out['out/in'].count())
df_in.reset_index(drop=True)
df_in = df_in.copy()
df_in['teste_data'] = pd.to_datetime(df_in['noted_date'], format="%d-%m-%Y %H:%M", errors='coerce')
df_in.info()
print(df_in.isnull().sum())
print(df_in.isna().sum())
