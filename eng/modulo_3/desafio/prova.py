import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import f1_score, make_scorer

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
'''
V1. Age of the patient. Any patient whose age exceeded 89 is listed as being of age "90".
V2. Gender of the patient
V3. Total Bilirubin
V4. Direct Bilirubin
V5. Alkphos Alkaline Phosphatase
V6. Sgpt Alanine Aminotransferase
V7. Sgot Aspartate Aminotransferase
V8. Total Proteins
V9. Albumin
V10. A/G Ratio Albumin and Globulin Ratio
'''
df = pd.read_csv('indian_liver_patient.csv', header=None,
                 names=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "Class"])

df['Gender'] = df['V2'].map({'Female': 0, 'Male': 1})
df.drop(columns=['V2'], inplace=True)
print(df.head())
df.fillna(df.mean(), inplace=True)
# Separar features e target
X = df.drop('Class', axis=1)
y = df['Class']
svc_param_grid = {
    'kernel': ['sigmoid', 'poly', 'rbf'],
    'C': uniform(1, 10)
}
kf = StratifiedKFold(n_splits=10, random_state=5762, shuffle=True)
classifier = SVC()
f1 = make_scorer(f1_score)
svc = SVC(random_state=5762)
svc_random_search = RandomizedSearchCV(estimator=svc, param_distributions=svc_param_grid, n_iter=5, scoring=f1, cv=kf, random_state=5762)
svc_random_search.fit(X, y)
print('Melhor resultado f1:', svc_random_search.best_score_)
print('\n\nMelhor configuração de hiperparâmetros:', svc_random_search.best_params_)
print( '\n\nConfigurações de todos os hiperparâmetros do melhor estimado encontrado pelo RandomizedSearchCV: \n', svc_random_search.best_estimator_)