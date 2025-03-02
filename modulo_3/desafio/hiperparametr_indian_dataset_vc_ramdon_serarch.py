import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

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

#print(df.columns)
X = df.drop('Class', axis=1)
y = df['Class']

#definindo hiperparâmetros
hiperparam = {'kernel':('sigmoid', 'rbf'), 'C':[0.01, 1, 10]}
#definindo o tipo de validacao cruzada e o numero de folds
cv_strat = StratifiedKFold(n_splits = 10)
#instânciando meu classificador
classifier = SVC()
#definindo a estrategia de score a partir da metrica f1
f1 = make_scorer(f1_score)
#instanciando e modelando o grid search com os hiperparametros e a validação definidas.
grid_cv = GridSearchCV(classifier, hiperparam, cv = cv_strat, scoring = f1)
print(grid_cv.fit(X, y))
#vamos olhar para todos os resultados encontrados!
print('Resumo de todos os resultados encontrados:\n\n', grid_cv.cv_results_)

#vamos olhar para os melhores resultados encontrados pelo Grid Search
print('Melhor resultado f1:', grid_cv.best_score_)
print('\n\nMelhor configuração de hiperparâmetros:', grid_cv.best_params_)

print( '\n\nConfigurações de todos os hiperparâmetros do melhor estimado encontrado pelo GridSearch: \n', grid_cv.best_estimator_)

print('------------- Vamos agora repetir o processo para o Random -------------------')
#definindo hiperparâmetros
#hiperparam1 = {'n_estimators':[10, 100, 1000]}
hiperparam1 = {'n_estimators':[10, 100, 1000], 'bootstrap': (True, False)}
#hiperparam1 = {'n_estimators':[10, 100, 1000], 'bootstrap': (True, False), 'criterion': ('gini', 'entropy')}

#instânciando meu classificador
classifier1 = RandomForestClassifier()

#instanciando e modelando o grid search com os hiperparametros e a validação definidas.
grid_cv1 = GridSearchCV(classifier1, hiperparam1, cv = cv_strat, scoring = f1)
grid_cv1.fit(X, y)
#vamos olhar para todos os resultados encontrados!
print('Resumo de todos os resultados encontrados:\n\n', grid_cv1.cv_results_)
#vamos olhar para os melhores resultados encontrados pelo Grid Search
print('Melhor resultado f1:', grid_cv1.best_score_)
print('\n\nMelhor configuração de hiperparâmetros:', grid_cv1.best_params_)
print( '\n\nConfigurações de todos os hiperparâmetros do melhor estimado encontrado pelo GridSearch: \n', grid_cv1.best_estimator_)