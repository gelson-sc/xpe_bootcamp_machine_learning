import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re

df = pd.read_csv('corpus.csv', encoding='latin1')
print(df.head())
print(df.info())
print(df.describe())
# Verificando a distribuição das classes
print(df['label'].value_counts())

# Preparação dos dados
# Removendo linhas com valores ausentes
#df.dropna(inplace=True)
# Convertendo o texto para minúsculas
#df['text'] = df['text'].str.lower()
# Removendo pontuações
# df['text'] = df['text'].str.replace('[^\w\s]', '')
# Download de recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Função para pré-processamento de texto
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenização
    tokens = word_tokenize(text)
    # Remover stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lematização
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Juntar tokens de volta em uma string
    return ' '.join(tokens)


# Aplicar pré-processamento
df['processed_text'] = df['text'].apply(preprocess_text)

# Verificar resultado do pré-processamento
print(df[['text', 'processed_text']].head())
exit(0)
# Passo 4: Vetorização do texto
# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['label'], test_size=0.2, random_state=42
)

# Vetorização com Bag of Words
count_vectorizer = CountVectorizer(max_features=5000)
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Vetorização com TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Passo 5: Treinar e avaliar classificador Naive Bayes
# Treinar classificador Naive Bayes com vetorização Bag of Words
nb_counts = MultinomialNB()
nb_counts.fit(X_train_counts, y_train)
y_pred_nb_counts = nb_counts.predict(X_test_counts)

# Avaliar o modelo
print("Naive Bayes com Bag of Words:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_nb_counts):.4f}")
print(classification_report(y_test, y_pred_nb_counts))

# Treinar classificador Naive Bayes com vetorização TF-IDF
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
y_pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)

# Avaliar o modelo
print("\nNaive Bayes com TF-IDF:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_nb_tfidf):.4f}")
print(classification_report(y_test, y_pred_nb_tfidf))

# 6 Treinar e avaliar classificador SVM
# Treinar classificador SVM com vetorização Bag of Words
svm_counts = SVC(kernel='linear')
svm_counts.fit(X_train_counts, y_train)
y_pred_svm_counts = svm_counts.predict(X_test_counts)

# Avaliar o modelo
print("SVM com Bag of Words:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_svm_counts):.4f}")
print(classification_report(y_test, y_pred_svm_counts))

# Treinar classificador SVM com vetorização TF-IDF
svm_tfidf = SVC(kernel='linear')
svm_tfidf.fit(X_train_tfidf, y_train)
y_pred_svm_tfidf = svm_tfidf.predict(X_test_tfidf)

# Avaliar o modelo
print("\nSVM com TF-IDF:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_svm_tfidf):.4f}")
print(classification_report(y_test, y_pred_svm_tfidf))

# 7
# Treinar e avaliar classificador Random Forest
# Treinar classificador Random Forest com vetorização Bag of Words
rf_counts = RandomForestClassifier(n_estimators=100, random_state=42)
rf_counts.fit(X_train_counts, y_train)
y_pred_rf_counts = rf_counts.predict(X_test_counts)

# Avaliar o modelo
print("Random Forest com Bag of Words:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf_counts):.4f}")
print(classification_report(y_test, y_pred_rf_counts))

# Treinar classificador Random Forest com vetorização TF-IDF
rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_tfidf.fit(X_train_tfidf, y_train)
y_pred_rf_tfidf = rf_tfidf.predict(X_test_tfidf)

# Avaliar o modelo
print("\nRandom Forest com TF-IDF:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf_tfidf):.4f}")
print(classification_report(y_test, y_pred_rf_tfidf))

#8
# Função para plotar matriz de confusão
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Previsto')
    plt.show()

# Plotar matrizes de confusão
plot_confusion_matrix(y_test, y_pred_nb_tfidf, 'Matriz de Confusão - Naive Bayes (TF-IDF)')
plot_confusion_matrix(y_test, y_pred_svm_tfidf, 'Matriz de Confusão - SVM (TF-IDF)')
plot_confusion_matrix(y_test, y_pred_rf_tfidf, 'Matriz de Confusão - Random Forest (TF-IDF)')

# Comparação dos modelos
modelos = ['NB (BoW)', 'NB (TF-IDF)', 'SVM (BoW)', 'SVM (TF-IDF)', 'RF (BoW)', 'RF (TF-IDF)']
acuracias = [
    accuracy_score(y_test, y_pred_nb_counts),
    accuracy_score(y_test, y_pred_nb_tfidf),
    accuracy_score(y_test, y_pred_svm_counts),
    accuracy_score(y_test, y_pred_svm_tfidf),
    accuracy_score(y_test, y_pred_rf_counts),
    accuracy_score(y_test, y_pred_rf_tfidf)
]

plt.figure(figsize=(12, 6))
sns.barplot(x=modelos, y=acuracias)
plt.title('Comparação da Acurácia dos Modelos')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()