import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
# Carregar os dados
df = pd.read_csv('winequality-red.csv', sep=';')

# Separar features e target
X = df.drop('quality', axis=1)  # Todas as colunas, exceto 'quality'
y = df['quality']  # Coluna 'quality' como target

# Dividir os dados (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Normalizar os dados com MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Função para treinar e avaliar modelos
def train_and_evaluate(model, model_name):
    print(f"\n--- {model_name} ---")
    # Treinar o modelo
    model.fit(X_train, y_train)
    # Fazer previsões
    y_pred = model.predict(X_test)
    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.2f}")
    # print("\nRelatório de Classificação:")
    # print(classification_report(y_test, y_pred))
    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão:")
    print(conf_matrix)
    # Visualizar a matriz de confusão
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    # plt.xlabel('Previsão')
    # plt.ylabel('Real')
    # plt.title(f'Matriz de Confusão - {model_name}')
    # plt.show()


# 1. KNN
knn = KNeighborsClassifier(n_neighbors=5)
train_and_evaluate(knn, "KNN")

# 2. DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=1)
train_and_evaluate(dtree, "DecisionTreeClassifier")

# 3. RandomForestClassifier
rf = RandomForestClassifier(max_depth=10, random_state=1)
train_and_evaluate(rf, "RandomForestClassifier")

# 4. SVM
svm = SVC(kernel='rbf', random_state=1)
train_and_evaluate(svm, "SVM")

# 5. MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=1)
train_and_evaluate(mlp, "MLPClassifier")
