import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

#df = pd.read_csv("bikeshare.csv")
df = pd.read_csv("hour.csv")
#df = pd.read_csv("day.csv")
# print(df.head(10))
# print(df.describe())
# print(df.info())
# print(df.shape)
# print(df.isnull().sum())
# contas quanto stem de 2012
df['dteday'] = pd.to_datetime(df['dteday'])
ano = 2012
qtde_registros_ano = df[df['dteday'].dt.year == ano]
quantidade_registros_ano = len(qtde_registros_ano)
print(f"Quantidade de registros em {ano}: {quantidade_registros_ano}")

#Quantas locações de bicicletas foram efetuadas em 2011?
soma_bike_ano = qtde_registros_ano['registered'].sum()
print(f'locações de bicicletas ({ano}): {soma_bike_ano}')

#Qual estação do ano contém a maior média de locações de bicicletas?
# Agrupando por 'season' e calculando a média de 'registered'
media_por_estacao = df.groupby('season')['registered'].mean().reset_index()
# Exibindo a média de locações por estação
print(media_por_estacao)