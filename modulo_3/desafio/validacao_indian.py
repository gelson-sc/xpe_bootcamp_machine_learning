import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
df = pd.read_csv('processed_indian_liver_patient.csv')
print(df.head(10))
# print(df.describe())
# print(df.shape)
# print(df.isnull().sum())
#{'Female': 0, 'Male': 1}
class_counts = df['Class'].value_counts()
print("Contagem de valores na coluna 'Class' # 1 -> Vivos 2 -> Mortos:")
print(class_counts)
# # Passo 2: Verificar dados categóricos e valores faltantes
# print("\nVerificação de dados categóricos e valores faltantes:")
# print(df.info())
# print("\nValores ausentes por coluna:")
# print(df.isna().sum())
X = df.drop('Class', axis=1)
y = df['Class']
print(X.head(3))
print(y.head(3))
print('Amostras e Features:', df.shape)

print(df.columns)

name_to_class = {
    'vivo': 1,
    'morto': 2
}
# data = np.array(X)
# print(data)
train_data1, test_data1, train_labels1, test_labels1 = train_test_split(X, y, test_size = 0.25, random_state = 42)
train_data2, test_data2, train_labels2, test_labels2 = train_test_split(X, y, test_size = 0.25, random_state = 123)
train_data3, test_data3, train_labels3, test_labels3 = train_test_split(X, y, test_size = 0.35, random_state = 42)

# treinando o modelo
classifier1 = RandomForestClassifier(n_estimators= 10, random_state=42).fit(train_data1, train_labels1)
classifier2 = RandomForestClassifier(n_estimators= 10, random_state=42).fit(train_data2, train_labels2)
classifier3 = RandomForestClassifier(n_estimators= 10, random_state=42).fit(train_data3, train_labels3)
predictions1_labels = classifier1.predict(test_data1)
predictions2_labels = classifier2.predict(test_data2)
predictions3_labels = classifier3.predict(test_data3)
# Exibindo dataframe com valores 10 reais e suas respectivas previsões
p = pd.DataFrame({'Real': test_labels3, 'Previsto': predictions3_labels})
# print(p.head(10))
# print(predictions1_labels)
# print(predictions2_labels)

print('\nAcurácia 1\n', metrics.accuracy_score(test_labels1, predictions1_labels))
print('\nAcurácia 2\n', metrics.accuracy_score(test_labels2, predictions2_labels))
print('\nAcurácia 3\n', metrics.accuracy_score(test_labels3, predictions3_labels))


#avaliando o modelo
print('Matriz de Confusão 1\n', metrics.confusion_matrix(test_labels1, predictions1_labels))
print('Matriz de Confusão 2\n', metrics.confusion_matrix(test_labels2, predictions2_labels))
print('Matriz de Confusão 3\n', metrics.confusion_matrix(test_labels3, predictions3_labels))

print('\nAcurácia Balanceada por classe 1\n', metrics.balanced_accuracy_score(test_labels1, predictions1_labels))
print('\nAcurácia Balanceada por classe 2\n', metrics.balanced_accuracy_score(test_labels2, predictions2_labels))
print('\nAcurácia Balanceada por classe 3\n', metrics.balanced_accuracy_score(test_labels3, predictions3_labels))

print('\nPrecision 1\n', metrics.precision_score(test_labels1, predictions1_labels))
print('\nPrecision 2\n', metrics.precision_score(test_labels2, predictions2_labels))
print('\nPrecisio 3n\n', metrics.precision_score(test_labels3, predictions3_labels))

print('\nRecall 1\n', metrics.recall_score(test_labels1, predictions1_labels))
print('\nRecall 2\n', metrics.recall_score(test_labels2, predictions2_labels))
print('\nRecall 3\n', metrics.recall_score(test_labels3, predictions3_labels))

print('\nF1\n', metrics.f1_score(test_labels1, predictions1_labels))
print('\nF1\n', metrics.f1_score(test_labels2, predictions2_labels))
print('\nF1\n', metrics.f1_score(test_labels3, predictions3_labels))

print('\nAUCROC\n', metrics.roc_auc_score(test_labels1, predictions1_labels))
print('\nAUCROC\n', metrics.roc_auc_score(test_labels2, predictions2_labels))
print('\nAUCROC\n', metrics.roc_auc_score(test_labels3, predictions3_labels))

print('\nClassification Report 1\n', metrics.classification_report(test_labels1, predictions1_labels))
print('\nClassification Report 2\n', metrics.classification_report(test_labels2, predictions2_labels))
print('\nClassification Report 3\n', metrics.classification_report(test_labels3, predictions3_labels))