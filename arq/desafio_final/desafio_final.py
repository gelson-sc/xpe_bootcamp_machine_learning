import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', 1000)
df = pd.read_csv('/home/gelson/datasets/creditcard.csv')
print(df.head())
print(df.info())