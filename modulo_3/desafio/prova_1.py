import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import f1_score, make_scorer


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('indian_liver_patient.csv', header=None,
                 names=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "Class"])

df['Gender'] = df['V2'].map({'Female': 0, 'Male': 1})
df.drop(columns=['V2'], inplace=True)
df.fillna(df.mean(), inplace=True)
X = df.drop('Class', axis=1)
y = df['Class']

cv_strat = StratifiedKFold(n_splits = 10)
f1 = make_scorer(f1_score)
distributions = dict(kernel = ['sigmoid', 'rbf', 'poly'],
                     C = uniform(loc=0, scale=10))
classifier = SVC()
random_cv = RandomizedSearchCV(classifier, distributions, cv = cv_strat, scoring = f1, random_state = 42, n_iter = 5)
random_cv.fit(X, y)
print('Resumo de todos os resultados encontrados:\n\n', random_cv.cv_results_)
#vamos olhar para os melhores resultados encontrados pelo Grid Search
print('Melhor resultado f1:', random_cv.best_score_)
print('\n\nMelhor configuração de hiperparâmetros:', random_cv.best_params_)
print( '\n\nConfigurações de todos os hiperparâmetros do melhor estimado encontrado pelo GridSearch: \n', random_cv.best_estimator_)