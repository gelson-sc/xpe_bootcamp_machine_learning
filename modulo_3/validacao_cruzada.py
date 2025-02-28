from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Carregar dataset
data = load_iris()
#print(data.DESCR)
#print(data)
X, y = data.data, data.target
print(X)
print(y)
# Criar modelo
model = RandomForestClassifier()

# Validação cruzada com 5 folds
scores = cross_val_score(model, X, y, cv=5)
print(scores)
print("Acurácia média: {:.2f}%".format(scores.mean() * 100))
