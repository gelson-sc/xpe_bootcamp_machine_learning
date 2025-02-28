import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
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
print(df.head(10))
print(df.describe())
print(df.shape)
print(df.isnull().sum())
class_counts = df['Class'].value_counts()
print("Contagem de valores na coluna 'Class' # 1 -> Vivos 2 -> Mortos:")
print(class_counts)
# Passo 2: Verificar dados categóricos e valores faltantes
print("\nVerificação de dados categóricos e valores faltantes:")
print(df.info())
print("\nValores ausentes por coluna:")
print(df.isna().sum())

# Passo 3: Mapear a feature 'Gender'
df['Gender'] = df['V2'].map({'Female': 0, 'Male': 1})
df.drop(columns=['V2'], inplace=True)
print(df.head())
# Tratar valores faltantes (opcional, dependendo da análise anterior)
df.fillna(df.mean(), inplace=True)
# Salvar o dataset processado
df.to_csv('processed_indian_liver_patient.csv', index=False)

# Separar features e target
X = df.drop('Class', axis=1)
y = df['Class']

# Definir métrica de avaliação
f1 = make_scorer(f1_score)

# Configuração do Kfold estratificado
kf = StratifiedKFold(n_splits=10, random_state=5762, shuffle=True)

# Parâmetros para SVC
svc_param_grid = {
    'kernel': ['sigmoid', 'poly', 'rbf'],
    'C': uniform(1, 10)
}

# Modelagem com SVC
svc = SVC(random_state=5762)
svc_random_search = RandomizedSearchCV(estimator=svc, param_distributions=svc_param_grid, n_iter=5, scoring=f1, cv=kf, random_state=5762)
svc_random_search.fit(X, y)

# Resultados do SVC
print("SVC Best Score:", svc_random_search.best_score_)
print("SVC Best Params:", svc_random_search.best_params_)
print("SVC Best Estimator:", svc_random_search.best_estimator_)

# Parâmetros para Random Forest
rf_param_grid = {
    'n_estimators': randint(10, 1000),
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Modelagem com Random Forest
rf = RandomForestClassifier(random_state=5762)
rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_grid, n_iter=5, scoring=f1, cv=kf, random_state=5762)
rf_random_search.fit(X, y)

# Resultados do Random Forest
print("Random Forest Best Score:", rf_random_search.best_score_)
print("Random Forest Best Params:", rf_random_search.best_params_)
print("Random Forest Best Estimator:", rf_random_search.best_estimator_)