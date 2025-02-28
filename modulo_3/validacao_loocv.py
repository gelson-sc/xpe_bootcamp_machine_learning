from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

modelo = LogisticRegression(max_iter=1000)

# Configurar LOOCV
loo = LeaveOneOut()
# Executar a validação cruzada
scores = cross_val_score(modelo, X, y, cv=loo)

# Resultados
print(f"Número de folds em LOOCV: {len(scores)}")
print(f"Acurácia média: {np.mean(scores):.4f}")
print(f"Desvio padrão: {np.std(scores):.4f}")
# Implementação manual de LOOCV
scores_manual = []

for train_idx, test_idx in loo.split(X):
    # Separação dos dados
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Treinamento
    modelo.fit(X_train, y_train)

    # Predição e avaliação
    y_pred = modelo.predict(X_test)
    scores_manual.append(accuracy_score(y_test, y_pred))

#print(scores_manual)
print(f"Acurácia média (implementação manual): {np.mean(scores_manual):.4f}")