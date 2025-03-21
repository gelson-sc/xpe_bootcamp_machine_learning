import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head(10))
print(df.tail(10))
print(df.describe())
print(df.info())
print(df.shape)
print(df.isnull().sum())
# coluna de saida quality
#print(df['quality'].sum())
exit(0)
# Separar features e target
X = df.drop('quality', axis=1)  # Todas as colunas, exceto 'quality'
y = df['quality']  # Coluna 'quality' como target

# Dividir os dados (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Normalizar os dados com MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dicionário para armazenar os modelos e suas acurácias
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(max_depth=10, random_state=1),
    "SVM": SVC(kernel='rbf', random_state=1, gamma='auto'),
    "MLP": MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 5), max_iter=1000, random_state=1)
}

# Treinar e avaliar os modelos
best_accuracy = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    print(f"\n--- Treinando {name} ---")
    # Treinar o modelo
    model.fit(X_train, y_train)
    # Fazer previsões
    y_pred = model.predict(X_test)
    # Calcular acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.2f}")
    # print("\nRelatório de Classificação:")
    # print(classification_report(y_test, y_pred))
    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão:")
    print(conf_matrix)

    # Verificar se é o melhor modelo
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Salvar o melhor modelo
if best_model is not None:
    print(f"\nMelhor modelo: {best_model_name} com acurácia de {best_accuracy:.2f}")
    joblib.dump(best_model, 'melhor_modelo.pkl')
    joblib.dump(scaler, 'scaler.pkl')  # Salvar o scaler para uso futuro
    print("Modelo e scaler salvos com sucesso!")
else:
    print("Nenhum modelo foi treinado.")