import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("bikeshare.csv")
print(df.head(10))
print(df.describe())
print(df.info())
print(df.shape)
print(df.isnull().sum())
# contas quanto stem de 2012
df['datetime'] = pd.to_datetime(df['datetime'])
# Quantos registros existem para o ano de 2012?
# Filtrando os registros de 2011
# registros_2011 = df[df['datetime'].dt.year == 2011]
# # Contando os registros de 2011
# quantidade_registros_2011 = len(registros_2011)
# print(f"Quantidade de registros em 2011: {quantidade_registros_2011}")

# Extraindo o ano da coluna 'data'
# df['ano'] = df['datetime'].dt.year
# # Agrupando por ano e contando os registros
# contagem_por_ano = df.groupby('ano').size().reset_index(name='contagem')
# print(contagem_por_ano)
# # outro exemplo
# contagem_por_ano = df.groupby('ano')['datetime'].count().reset_index(name='contagem')
# print(contagem_por_ano)
#Quantas locações de bicicletas foram efetuadas em 2011?
# registros_2011 = df[df['datetime'].dt.year == 2011]
# print(registros_2011.head(10))
# soma_bike_2011 = registros_2011['registered'].sum()
# print(soma_bike_2011)

#Qual estação do ano contém a maior média de locações de bicicletas?
# Convertendo a coluna 'datetime' para datetime
# df['datetime'] = pd.to_datetime(df['datetime'])
# Agrupando por 'season' e calculando a média de 'registered'
# media_por_estacao = df.groupby('season')['registered'].mean().reset_index()
# # Exibindo a média de locações por estação
# print(media_por_estacao)
# # Encontrando a estação com a maior média
# estacao_maior_media = media_por_estacao.loc[media_por_estacao['registered'].idxmax()]
#
# print("\nEstação com a maior média de locações:")
# print(estacao_maior_media)
# mapeamento_estacoes = {1: 'Inverno', 2: 'Primavera', 3: 'Verão', 4: 'Outono'}

# Qual horário do dia contém a maior média de locações de bicicletas?
# df['hora'] = df['datetime'].dt.hour
# media_por_hora = df.groupby('hora')['registered'].mean().reset_index()
# print(media_por_hora)
# # Encontrando o horário com a maior média
# horario_maior_media = media_por_hora.loc[media_por_hora['registered'].idxmax()]
# print("\nHorário com a maior média de locações:")
# print(horario_maior_media)

# Que dia da semana contém a maior média de locações de bicicletas?
#  (0: domingo, 1: segunda-feira, …, 6: sábado)
df['dia_semana'] = df['datetime'].dt.dayofweek
print(df['dia_semana'].head())
media_por_dw = df.groupby('dia_semana')['registered'].mean().reset_index()
print(media_por_dw)
dw_maior_media = media_por_dw.loc[media_por_dw['registered'].idxmax()]
print(dw_maior_media)
