from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Definir os hiperparâmetros a serem testados
param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20)
}
model = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5)
random_search.fit(X_train, y_train)
# Melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", random_search.best_params_)

# Avaliar o modelo com os melhores hiperparâmetros
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia com melhores hiperparâmetros: {:.2f}%".format(accuracy * 100))