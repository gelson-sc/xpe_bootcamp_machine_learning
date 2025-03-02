import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
df = pd.read_csv('processed_indian_liver_patient.csv')
print(df.head(10))
print(df.describe())
class_counts = df['Class'].value_counts()
print("Contagem de valores na coluna 'Class' # 1 -> Vivos 2 -> Mortos:")
print(class_counts)
print('Amostras e Features:', df.shape)

print(df.columns)
X = df.drop('Class', axis=1)
y = df['Class']

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)
# treinando o SVM com kernel RBF
print('---------------------- KERNEL RBF ----------------------')
classifier1 = SVC(kernel='rbf').fit(train_data,train_labels)
# aplicando o modelo treinado
predictions1_labels = classifier1.predict(test_data)

# Exibindo dataframe com valores 10 reais e suas respectivas previsões
p = pd.DataFrame({'Real': test_labels, 'Previsto': predictions1_labels})
print(p.head(10))

print('Matriz de Confusão\n', metrics.confusion_matrix(test_labels, predictions1_labels))
print('\nAcurácia\n', metrics.accuracy_score(test_labels, predictions1_labels))
print('\nF1\n', metrics.f1_score(test_labels, predictions1_labels))
print('\nAUCROC\n', metrics.roc_auc_score(test_labels, predictions1_labels))

# treinando o SVM com kernel sigmoidal
print('---------------------- KERNEL SIGMOIDAL ----------------------')
classifier2 = SVC(kernel='sigmoid').fit(train_data,train_labels)

# aplicando o modelo treinado
predictions2_labels = classifier2.predict(test_data)

# Exibindo dataframe com valores 10 reais e suas respectivas previsões
p = pd.DataFrame({'Real': test_labels, 'Previsto': predictions2_labels})
print(p.head(10))
print('Matriz de Confusão\n', metrics.confusion_matrix(test_labels, predictions2_labels))
print('\nAcurácia\n', metrics.accuracy_score(test_labels, predictions2_labels))
print('\nF1\n', metrics.f1_score(test_labels, predictions2_labels))
print('\nAUCROC\n', metrics.roc_auc_score(test_labels, predictions2_labels))

print('---------------------- KERNEL GAMA ----------------------')
# treinando o SVM com kernel rbf, com largura da gaussiana alterada
classifier3 = SVC(kernel='rbf', gamma = 'auto').fit(train_data,train_labels)

# aplicando o modelo treinado
predictions3_labels = classifier3.predict(test_data)

# Exibindo dataframe com valores 10 reais e suas respectivas previsões
p = pd.DataFrame({'Real': test_labels, 'Previsto': predictions3_labels})
print(p.head(10))
print('Matriz de Confusão\n', metrics.confusion_matrix(test_labels, predictions3_labels))
print('\nAcurácia\n', metrics.accuracy_score(test_labels, predictions3_labels))
print('\nF1\n', metrics.f1_score(test_labels, predictions3_labels))
print('\nAUCROC\n', metrics.roc_auc_score(test_labels, predictions3_labels))

print('------------- kernel=RBF, gamma = 0.0001--------------------')
# treinando o SVM com kernel rbf, com largura da gaussiana alterada
classifier4 = SVC(C = 0.1, kernel='rbf', gamma = 0.0001).fit(train_data,train_labels)
predictions4_labels = classifier4.predict(test_data)
# Exibindo dataframe com valores 10 reais e suas respectivas previsões
p = pd.DataFrame({'Real': test_labels, 'Previsto': predictions4_labels})
p.head(10)
print('Matriz de Confusão\n', metrics.confusion_matrix(test_labels, predictions4_labels))
print('\nAcurácia\n', metrics.accuracy_score(test_labels, predictions4_labels))
print('\nF1\n', metrics.f1_score(test_labels, predictions4_labels))
print('\nAUCROC\n', metrics.roc_auc_score(test_labels, predictions4_labels))

print('-------------- Random Forest Classifiers ---------------------')
# treinando o modelo
classifier5 = RandomForestClassifier(n_estimators= 10, random_state=42).fit(train_data, train_labels);
# aplicando o modelo treinado para a previsão do resultado do teste
predictions5_labels = classifier5.predict(test_data)
# Exibindo dataframe com valores 10 reais e suas respectivas previsões
p = pd.DataFrame({'Real': test_labels, 'Previsto': predictions5_labels})
print(p.head(10))
#avaliando o modelo
print('Matriz de Confusão\n', metrics.confusion_matrix(test_labels, predictions5_labels))
print('\nAcurácia\n', metrics.accuracy_score(test_labels, predictions5_labels))
print('\nF1\n', metrics.f1_score(test_labels, predictions5_labels))
print('\nAUCROC\n', metrics.roc_auc_score(test_labels, predictions5_labels))
print('----------------- TROCA RENDOM STATE PARA 124 --------------------')
# treinando o modelo
classifier6 = RandomForestClassifier(n_estimators= 10, random_state = 124).fit(train_data, train_labels);
# aplicando o modelo treinado para a previsão do resultado do teste
predictions6_labels = classifier6.predict(test_data)
# Exibindo dataframe com valores 10 reais e suas respectivas previsões
p = pd.DataFrame({'Real': test_labels, 'Previsto': predictions5_labels})
print(p.head(10))
#avaliando o modelo
print('Matriz de Confusão\n', metrics.confusion_matrix(test_labels, predictions6_labels))
print('\nAcurácia\n', metrics.accuracy_score(test_labels, predictions6_labels))
print('\nF1\n', metrics.f1_score(test_labels, predictions6_labels))
print('\nAUCROC\n', metrics.roc_auc_score(test_labels, predictions6_labels))

print('--------------------- aumentar o número de estimadores ------------------')
# treinando o modelo
classifier7 = RandomForestClassifier(n_estimators= 100, random_state = 42).fit(train_data, train_labels);
# aplicando o modelo treinado para a previsão do resultado do teste
predictions7_labels = classifier7.predict(test_data)
# Exibindo dataframe com valores 10 reais e suas respectivas previsões
p = pd.DataFrame({'Real': test_labels, 'Previsto': predictions5_labels})
print(p.head(10))
print('Matriz de Confusão\n', metrics.confusion_matrix(test_labels, predictions7_labels))
print('\nAcurácia\n', metrics.accuracy_score(test_labels, predictions7_labels))
print('\nF1\n', metrics.f1_score(test_labels, predictions7_labels))
print('\nAUCROC\n', metrics.roc_auc_score(test_labels, predictions7_labels))

print('----------------- bootstrap true -----------------------')
# treinando o modelo
classifier8 = RandomForestClassifier(n_estimators= 100, bootstrap = False, random_state = 42).fit(train_data, train_labels);
# aplicando o modelo treinado para a previsão do resultado do teste
predictions8_labels = classifier8.predict(test_data)
# Exibindo dataframe com valores 10 reais e suas respectivas previsões
p = pd.DataFrame({'Real': test_labels, 'Previsto': predictions8_labels})
print(p.head(10))
print('Matriz de Confusão\n', metrics.confusion_matrix(test_labels, predictions8_labels))
print('\nAcurácia\n', metrics.accuracy_score(test_labels, predictions8_labels))
print('\nF1\n', metrics.f1_score(test_labels, predictions8_labels))
print('\nAUCROC\n', metrics.roc_auc_score(test_labels, predictions8_labels))