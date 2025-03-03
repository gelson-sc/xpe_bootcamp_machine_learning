import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import uniform
from scipy.stats import randint

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

#definindo o tipo de validacao cruzada e o numero de folds
cv_strat = StratifiedKFold(n_splits = 10)
#instânciando meu classificador
classifier = SVC()
#definindo a estrategia de score a partir da metrica f1
f1 = make_scorer(f1_score)
#definindo hiperparâmetros
distributions = dict(kernel = ['sigmoid', 'rbf'], C = uniform(loc=0, scale=10))
#instanciando e modelando o grid search com os hiperparametros e a validação definidas.
random_cv = RandomizedSearchCV(classifier, distributions, cv = cv_strat, scoring = f1, random_state = 42, n_iter = 10)
random_cv.fit(X, y)
print('Resumo de todos os resultados encontrados:\n\n', random_cv.cv_results_)
print('Melhor resultado f1:', random_cv.best_score_)
print('\n\nMelhor configuração de hiperparâmetros:', random_cv.best_params_)
print( '\n\nConfigurações de todos os hiperparâmetros do melhor estimado encontrado pelo GridSearch: \n', random_cv.best_estimator_)

print('----------- Vamos testar agora o mesmo processo para o RF ------')
#definindo hiperparâmetros
distributions1 = dict(n_estimators = randint(10, 100),
                      bootstrap = [True, False],
                      criterion = ['gini', 'entropy'])
#instânciando meu classificador
classifier1 = RandomForestClassifier(random_state = 42)

#instanciando e modelando o grid search com os hiperparametros e a validação definidas.
random_cv1 = RandomizedSearchCV(classifier1, distributions1, cv = cv_strat, scoring = f1, random_state = 42, n_iter = 10)
random_cv1.fit(X, y)
#vamos olhar para todos os resultados encontrados!
print('Resumo de todos os resultados encontrados:\n\n', random_cv1.cv_results_)
#vamos olhar para os melhores resultados encontrados pelo Grid Search
print('Melhor resultado f1:', random_cv1.best_score_)
print('\n\nMelhor configuração de hiperparâmetros:', random_cv1.best_params_)
print( '\n\nConfigurações de todos os hiperparâmetros do melhor estimado encontrado pelo GridSearch: \n', random_cv1.best_estimator_)