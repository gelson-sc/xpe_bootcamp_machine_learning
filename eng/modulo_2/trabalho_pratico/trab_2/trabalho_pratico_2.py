import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("data_banknote_authentication.txt",
                 header=None, names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])
print(df.head(10))
print(df.describe())
print(df.info())
print(df.shape)
print(df.isnull().sum())
