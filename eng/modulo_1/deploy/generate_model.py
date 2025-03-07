#importando as bibliotecas
from sklearn import neighbors, datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
#salvando o modelo
import joblib
import math


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
#cria a rotina para utilizar o dataset Iris
iris = datasets.load_iris()
# print(iris)
#Converte o banco de dados iris para o dataframe
df_iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(df_iris.head())

#transforma os dados em array
X = df_iris.iloc[:, :-1].values  #dados de entrada
y = df_iris.iloc[:, 4].values  # saídas ou target
# print(X)
# print(y)

#realiza a divisão dos dados entre treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)# divide 20% para teste
# realiza o processo de normalização dos dados
scaler = StandardScaler()  #objeto que normaliza os dados
scaler.fit(X_train)  #realiza a normalização dos dados

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# print(X_train)
# print(X_test)
#treina o modelo
classifier = KNeighborsClassifier(n_neighbors=5) #utiliza a construção por meio de 5 vizinhos
clsf = classifier.fit(X_train, y_train) # aplica a classificação
print(clsf)
#realiza a previsão
y_pred = classifier.predict(X_test)
print(y_pred)
#constroi a matriz de confusão para comparar o modelo criado
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#salva o modelo
#salvando o modelo em disco
print(scaler.mean_)
print(scaler.var_)

print([math.sqrt(valores) for valores in scaler.var_]) #desvio padrao
exit(0)
nome_arquivo = 'modelo_knn_update.joblib'
joblib.dump(classifier, nome_arquivo)