import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('corpus.csv', encoding='latin1')
# Separar features e target
X = df['text']
y = df['label']

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Criar um vetorizador de texto
vetorizador = TfidfVectorizer()

# Transformar os dados em vetores
X_train_vet = vetorizador.fit_transform(X_train)
X_test_vet = vetorizador.transform(X_test)

# Treinar o modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_train_vet, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test_vet)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))