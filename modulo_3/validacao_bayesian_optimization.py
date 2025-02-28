from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Definir os hiperparâmetros a serem testados
param_space = {
    'n_estimators': Integer(10, 200),
    'max_depth': Integer(1, 50),
    'min_samples_split': Integer(2, 20)
}
model = RandomForestClassifier()
# Criar o Bayes Search com validação cruzada
bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=10, cv=5)

# Treinar o modelo com Bayes Search
bayes_search.fit(X_train, y_train)

# Melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", bayes_search.best_params_)

# Avaliar o modelo com os melhores hiperparâmetros
best_model = bayes_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia com melhores hiperparâmetros: {:.2f}%".format(accuracy * 100))