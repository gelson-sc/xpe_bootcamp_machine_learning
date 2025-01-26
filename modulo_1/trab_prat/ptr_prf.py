import pandas as pd  #bibioteca responsável para o tratamento e limpeza dos dados
import numpy as np #biblioteca utilizada para o tratamento eficiente de dados numéricos
import datetime  #biblioteca utilizada para trabalhar com datas
from matplotlib import pyplot as plt  #plotar os gráficos
import seaborn as sns #plot de gráficos

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 20000)
pd.set_option('display.width', 1000)

df_consultas=pd.read_csv('KaggleV2-May-2016.csv')
print(df_consultas.head(5))
# print(df_consultas.tail())
# print("*"*30, 'INFO', "*"*30)
# print(df_consultas.info())
# print("*"*30, 'ISNA', "*"*30)
# print(df_consultas.isna().sum())
# print(df_consultas.shape)
print(df_consultas.describe())
comparecimentos = df_consultas['No-show'].value_counts() # Yes o paciente não compareceu / No o paciente compareceu
print(comparecimentos)
print(df_consultas['SMS_received'].value_counts()) # 0 o paciente não reccebeu / 1 o paciente recebeu
print(df_consultas['Diabetes'].value_counts()) # 0 o paciente não reccebeu / 1 o paciente recebeu
media_faltas = df_consultas['No-show'].value_counts()['No']/len(df_consultas)
print("media_faltas", media_faltas)

idade_media = df_consultas['Age'].mean()
print(idade_media)

#contando a quantidade de valores distintos em cada uma das colunas
for colunas in list(df_consultas.columns):
  print( "{0:25} {1}".format(colunas, df_consultas[colunas].nunique()))

localodades_distintas = df_consultas['Neighbourhood'].nunique()
print(localodades_distintas)