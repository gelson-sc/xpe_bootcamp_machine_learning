import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
# print(df.info())
# print(df.head())
# print(df.describe())
# print(df.isnull().sum())
d = df["ACTIVITY"].value_counts().sort_index()
#print(d)
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
dict = {0: 'Em pé', 1: 'Andando', 2: 'Sentado', 3: 'Caindo', 4: 'Cãibras', 5: 'Correndo'}
resp = list(dict.keys())
labels = list(dict.values())
sizes = [d[0], d[1], d[2], d[3], d[4], d[5]]
explode = (0, 0, 0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=90, counterclock=False, shadow=False)
ax1.axis('equal')
# plt.show()

# Create pivot_table
colum_names = ['TIME','SL','EEG','BP','HR','CIRCLUATION']
df_pivot_table = df.pivot_table(colum_names,
               ['ACTIVITY'], aggfunc='median')
print(df_pivot_table)

# Correlation matrix
tmp = df.drop('ACTIVITY', axis=1)
correlations = tmp.corr()
print(correlations)
# Plot figsize
fig, ax = plt.subplots(figsize=(15, 11))
# Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
# Generate Heat Map, allow annotations and place floats in map
sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
ax.set_xticklabels(
    colum_names,
    rotation=45,
    horizontalalignment='right'
)
ax.set_yticklabels(colum_names)
# plt.show()
# df_teste = df.replace({'ACTIVITY':{0:'Standing',1:'Walking',2:'Sitting',3:'Falling',4:'Cramps',5:'Running'}})
# print(df_teste.head())
# print(df.shape)
# print(df.shape)
# df.boxplot(column=['SL', 'EEG', 'BP', 'HR'])
# plt.show()
# df.hist(bins=20, figsize=(20,20), color='g')
Target = df['ACTIVITY']
Features = df[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']]

X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.3)
# sc = StandardScaler()
# sc = MinMaxScaler(feature_range=(0, 1))
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)
rfc = RandomForestClassifier(
    n_estimators=50,
    # min_samples_leaf=10,
    # min_samples_split=10
)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
