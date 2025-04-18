import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf  # importando o tensorflow para ser utilizado como backend do treinamento
from tensorflow.keras.models import Sequential, \
    load_model  # importando os modelos sequenciais e a função para carregar o modelo
from tensorflow.keras.layers import Dense, Dropout, LSTM  # importando as camadas Densa, Dropout e LSTM

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU use
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

# arquivo que contém os dados para teste do modelo (motor da aeronave sem falha)
arquivoTeste = 'PM_test.txt'
# contém os dados para treinamento do modelo (falha do motor da aeronave) 100 diferentes motores
arquivoTreinamento = "PM_train.txt"
# dados reais sobre a vida útil dos motores das aeronaves
arquivoVerdade = 'PM_truth.txt'

# (motor da aeronave sem falha)
dataTeste = pd.read_csv(arquivoTeste, sep=' ', header=None)
# (falha do motor da aeronave) 100 diferentes motores
dataTreinamento = pd.read_csv(arquivoTreinamento, delimiter=' ', header=None)
# realiza a leitura do banco de dados que contém o "tempo de vida útil" de cada motor em ciclos
dataVerdade = pd.read_csv(arquivoVerdade, sep=' ', header=None)

# print(dataTeste.head())
# print(dataTreinamento.head())
# print(dataVerdade.head())

# retirando os dados que não interessam
dataTreinamento = dataTreinamento.drop(dataTreinamento.columns[26:28], axis=1)
dataTeste = dataTeste.drop(dataTeste.columns[26:28], axis=1)
dataVerdade = dataVerdade.drop(dataVerdade.columns[1], axis=1)

# adicionando o cabeçalho ao dataset
dataTreinamento.columns = ["id", "cycle", "setting1", "setting2", "setting3", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
                           "s8", "s9",
                           "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"]
dataTeste.columns = ["id", "cycle", "setting1", "setting2", "setting3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8",
                     "s9",
                     "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"]
dataVerdade.columns = ["RLU"]  # remaining useful life (RUL) - tempo de vida útil
# 100 diferentes motores com 21 sensores em cada um dos motores
# print(dataTreinamento.head(5))
count_data = len(dataTreinamento.groupby("id").count())
print(count_data)
# print das 10 ultimas linhas do dataset
print(dataTreinamento.tail(10))
# 100 diferentes motores com 21 sensores em cada um dos motores
print(dataTeste.head(5))
# tempo de vida restante de cada um dos motores (por exemplo, o motor 1 tem mais 112 ciclos de vida)
print(dataVerdade.head(5))
# conhecendo os dados
print(dataTreinamento.describe().transpose())
# retirando alguns sensores para facilitar a análise
dataTreinamentoNew = dataTreinamento.drop(['s1', 's5', 's10', 's16', 's18', 's19', 'setting3'], axis=1)
dataTesteNew = dataTeste.drop(['s1', 's5', 's10', 's16', 's18', 's19', 'setting3'], axis=1)
# Gerando histograma para visualizar algumas variáveis
dataTreinamentoNew.hist(bins=50, figsize=(18, 16))
# plt.show()
# quantidade de ciclos de cada motor - (tempo de funcionamento)
cyclestrain = dataTreinamentoNew.groupby('id', as_index=False)['cycle'].max()
# quantidade de ciclos de cada motor - (tempo de funcionamento)
cyclestest = dataTesteNew.groupby('id', as_index=False)['cycle'].max()

# plot das figuras que contém a quantidade de ciclos (tempo de vida) para cada motor
fig = plt.figure(figsize=(16, 12))
fig.add_subplot(1, 2, 1)
bar_labels = list(cyclestrain['id'])
bars = plt.bar(list(cyclestrain['id']), cyclestrain['cycle'], color='red')
plt.ylim([0, 400])
plt.xlabel('Id', fontsize=16)
plt.ylabel('Templo de Funcionamente (ciclos)', fontsize=16)
plt.title('Max. Ciclos - Treinamento', fontsize=16)
plt.xticks(np.arange(min(bar_labels) - 1, max(bar_labels) - 1, 5.0), fontsize=12)
plt.yticks(fontsize=12)
fig.add_subplot(1, 2, 2)
bars = plt.bar(list(cyclestest['id']), cyclestest['cycle'], color='grey')
plt.ylim([0, 400])
plt.xlabel('Id', fontsize=16)
plt.ylabel('Templo de Funcionamente (ciclos)', fontsize=16)
plt.title('Max. Ciclos - Teste', fontsize=16)
plt.xticks(np.arange(min(bar_labels) - 1, max(bar_labels) - 1, 5.0), fontsize=12)
plt.yticks(fontsize=12)
# plt.show()
# cria o dataset que contém os valores máximos para cada motor
dataTreinamentoNew = pd.merge(dataTreinamentoNew, dataTreinamentoNew.groupby('id', as_index=False)['cycle'].max(),
                              how='left', on='id')
print(dataTreinamentoNew.head())
# renomeia as colunas que foram adicionadas
dataTreinamentoNew.rename(columns={"cycle_x": "cycles", "cycle_y": "maxcycles"}, inplace=True)
print(dataTreinamentoNew.head())
# cria a coluna que contém o "resto de vida" a cada ciclo do motor
dataTreinamentoNew['TTF'] = dataTreinamentoNew['maxcycles'] - dataTreinamentoNew['cycles']
print(dataTreinamentoNew.head())

# aplicando a normalização
scaler = MinMaxScaler()
# realizando uma cópia do dataset para selecionar os dados de entrada para o treinamento
treinaNormalizado = dataTreinamentoNew.copy()
# seleciona todas as linhas e as colunas de 2 até a ultima (cycles até s21)
for col in treinaNormalizado.columns[2:19]:
    if treinaNormalizado[col].dtype != 'float64':
        treinaNormalizado[col] = treinaNormalizado[col].astype('float64')
treinaNormalizado.iloc[:, 2:19] = scaler.fit_transform(treinaNormalizado.iloc[:, 2:19])

# realizando o mesmo procedimento para os dados de teste
testeNormalizado = dataTesteNew.copy()
for col in testeNormalizado.columns[2:19]:
    if testeNormalizado[col].dtype != 'float64':
        testeNormalizado[col] = testeNormalizado[col].astype('float64')
testeNormalizado.iloc[:, 2:19] = scaler.transform(testeNormalizado.iloc[:, 2:19])
testeNormalizado.head()

print('treino normalizado\n', treinaNormalizado.head())
print('treino info\n', treinaNormalizado.info())

# exibindo os dados de treinamento e teste
fig = plt.figure(figsize=(8, 8))
fig.add_subplot(1, 2, 1)
plt.plot(dataTreinamentoNew[dataTreinamentoNew.id == 1].s2)  # plota os dados do sensor 2 para o motor 1
plt.plot(dataTesteNew[dataTesteNew.id == 1].s2)  # plota os dados do sensor 2 para o motor 1
plt.legend(['Treinamento', 'Teste'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
plt.ylabel('Dataset Original')
fig.add_subplot(1, 2, 2)
plt.plot(treinaNormalizado[treinaNormalizado.id == 1].s2)  # plota os dados do sensor 2 para o motor 1
plt.plot(testeNormalizado[testeNormalizado.id == 1].s2)  # plota os dados do sensor 2 para o motor 1
plt.legend(['Treinamento Normalizado', 'Teste Normalizado'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand",
           borderaxespad=0)
plt.ylabel('Dataset Normalizado')


# plt.show()


# Encontrando os valores para falha em cada ciclo**
# %%
# função que encontra a fração correspondente ao tempo de vida do motor
def fracaoDoTTF(dfMotor, q):
    # encontra a fração, utilizando os valores de máximo TFF e mínimo TTF
    return (dfMotor.TTF[q] - dfMotor.TTF.min()) / float(dfMotor.TTF.max() - dfMotor.TTF.min())


# aplica a função para cada um dos 100 motores existentes
# lista auxiliar para o tempo de falha
fTTFz = []
# lista que contém a fração do tempo de falha para cada motor em cada ciclo
fTTF = []

# for utilizado para computar a fração do tempo de vida restante para cada um dos motores
# (no início a fração do tempo de vida é 1 e cai até 0 quando o motor falha)
# esse for cria um "iterable" entre o menor valor de id(1) até o maior valor (100)+1
for i in range(dataTreinamentoNew['id'].min(), dataTreinamentoNew['id'].max() + 1):
    dat = dataTreinamentoNew[dataTreinamentoNew.id == i]  # seleciona cada um dos id presentes no dataset
    dat = dat.reset_index(drop=True)
    for q in range(len(dat)):  # utilizado para aplicar a transformação em fração sobre cada um dos ciclos
        fTTFz = fracaoDoTTF(dat, q)  # aplica a função que "normaliza" os dados de tempo de vida
        fTTF.append(fTTFz)  # adiciona à lista de tempo restante de vida
treinaNormalizado['fTTF'] = fTTF
print(treinaNormalizado.head())
# **Aplicando o modelo de previsão de falhas**
# %%
# selecionando os dados de entrada a saída
# seleciona todas as linhas e as colunas de "cycles" até s21 para a entrada do treinamento
X_train = treinaNormalizado.values[:, 1:19]
# seleciona todas as linhas e a coluna de fTTF para a saída (target) do treinamento
Y_train = treinaNormalizado.values[:, 21]
# seleciona todas as linhas e as colunas de "cycles" até s21 para a entrada de teste
X_test = testeNormalizado.values[:, 1:19]
print(X_train)
print(Y_train)

# **Através desses dados históricos de funcionamento normal e de falhas,
# é possível prever quando ocorrerá a próxima falha do motor?**
# %%
# %%
# criando o modelo sequencial para a classificação
modeloMLP = Sequential()  # cria o objeto para o modelo sequencial
# cria a camada de entrada - na camada de entrada é necessário definir a dimensão da entrada
# para os nossos dados, utilizamos 18 entradas e essa camada que é completamente conectada (Dense)
# possui 6 neurônios
modeloMLP.add(Dense(6, input_dim=18, kernel_initializer='normal', activation='relu'))
modeloMLP.add(Dense(12, kernel_initializer='normal', activation='relu'))  # camada escondida
modeloMLP.add(Dropout(0.2))  # camada de dropout utilizada para reduzir o overfiting
# adiciona a camada de saída que contém apenas 1 neurônio (classificação binária)
modeloMLP.add(Dense(1, kernel_initializer='normal'))
# define qual deve ser a função perda utilizada e qual o otimizador a ser utilizado
modeloMLP.compile(loss='mean_squared_error', optimizer='adam')
print(modeloMLP.summary())
# **Realizando a previsão do modelo**
modeloMLP.fit(X_train, Y_train, epochs=50, batch_size=10, verbose=2)
previsao = modeloMLP.predict(X_test)

print(previsao.min(), previsao.max())

print(X_test)
# cria o novo dataset completo para o teste
# agrupa o dataset de teste inicial com os resultados obtidos
dataTesteNew = pd.merge(dataTesteNew, dataTesteNew.groupby('id', as_index=False)['cycle'].max(), how='left', on='id')
print(dataTesteNew.head())

# modifica os nomes para facilitar a análise
dataTesteNew.rename(columns={"cycle_x": "cycles", "cycle_y": "maxcycles"}, inplace=True)
dataTesteNew['score'] = previsao
print(dataTesteNew.head())


# função para retornar os valores em escala anterior
def totcycles(data):
    return (data['cycles'] / (1 - data['score']))


dataTesteNew['maxpredcycles'] = totcycles(dataTesteNew)  # aplica a função
print(dataTesteNew.head())


# função para encontrar o tempo de vida dos motores baseados nos dados de entrada
def RULfunction(data):
    return (data['maxpredcycles'] - data['maxcycles'])


dataTesteNew['RUL'] = RULfunction(dataTesteNew)  # aplica a função de transformação "retorno" dos dados
print(dataTesteNew.head())

# %%
t = dataTesteNew.columns == 'RUL'  # utilizado para encontrar a posição do dataset que corresponde ao valor RUL
ind = [i for i, x in enumerate(t) if x]  # neste cado a posição é a coluna 22

predictedRUL = []  # lista para os valores rpevisto de RUL para cada um dos motores

# for utilizado para percorrer cada um dos motores e encontrar o valor da previsão
for i in range(dataTesteNew["id"].min(), dataTesteNew["id"].max() + 1):
    # seleciona cada um dos id de motores e encontra qual é o RUL
    npredictedRUL = dataTesteNew[dataTesteNew.id == i].iloc[dataTesteNew[dataTesteNew.id == i].cycles.max() - 1, ind]
    predictedRUL.append(npredictedRUL)  # adiciona para a lista de valores previstos

predictedRUL[0:10]  # imprime os RUL para os 10 primeiros id (já no fortamo final)

# %%
# plot da figura comparativa entre os valores reais e previstos para a manutenção/troca dos motores
plt.figure(figsize=(16, 8))
plt.plot(dataVerdade)
plt.plot(predictedRUL)
plt.xlabel('Id Motor', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('RUL', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['RUL Real', 'RUL Previsto'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
plt.show()
print(predictedRUL)
