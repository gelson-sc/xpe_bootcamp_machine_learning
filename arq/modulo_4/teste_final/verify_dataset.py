import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.utils import to_categorical
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
'''
ACTIVITY Classificação da atividade realizada no momento da coleta dos dados.
TIME	 Tempo de monitoramento (timestamp ou duração da atividade).
SL	     Nível de açúcar no sangue (provavelmente glicose).
EEG	     Taxa de atividade elétrica cerebral (eletroencefalograma).
BP	     Pressão arterial.
HR	     Frequência cardíaca (batimentos por minuto).
CIRCULATION	 Circulação sanguínea (indicador geral de fluxo ou saúde circulatória).

ACTIVITY

  Código	Atividade
0	Em pé (Standing)
1	Andando (Walking)
2	Sentado (Sitting)
3	Caindo (Falling)
4	Cãibras (Cramps)
5	Correndo (Running)

'''

compareScore = []
df = pd.read_csv('fall_detection.csv')
print(df.info())
print(df.head())
print(df.describe())
print(df.isnull().sum())
acumulados = df.groupby("ACTIVITY").count()['SL']
print(acumulados)

# df_teste = df.replace({'ACTIVITY':{0:'Standing',1:'Walking',2:'Sitting',3:'Falling',4:'Cramps',5:'Running'}})
# print(df_teste.head())
print(df.shape)