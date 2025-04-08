# Amazon Reviews Classification
# Classificação de textos de reviews da Amazon usando técnicas de NLP e Machine Learning

# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
import warnings

#warnings.filterwarnings('ignore')

# Download dos módulos necessários do NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Importação das ferramentas de processamento de texto do NLTK
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Carregamento dos dados
#df = pd.read_csv('corpus.csv', encoding='latin1')
df = pd.read_csv('https://raw.githubusercontent.com/Gunjitbedi/Text-Classification/master/corpus.csv',encoding='latin-1')

# Exibição das primeiras linhas do dataset
print("Informações sobre o dataset:")
print(f"Número de registros: {df.shape[0]}")
print(f"Número de colunas: {df.shape[1]}")
print("\nPrimeiras 5 linhas:")
print(df.head())

# Verificação da distribuição das classes
print("\nDistribuição dos rótulos:")
print(df['label'].value_counts())


# Função para pré-processamento de texto
def preprocess_text(text):
    if isinstance(text, str):
        # Conversão para minúsculas
        text = text.lower()

        # Remoção de caracteres especiais e números
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenização
        tokens = word_tokenize(text)

        # Remoção de stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Lematização
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Junção dos tokens em uma string
        return ' '.join(tokens)
    else:
        return ''


# Aplicação do pré-processamento aos textos
print("\nAplicando pré-processamento aos textos...")
df['processed_text'] = df['text'].apply(preprocess_text)

# Divisão dos dados em treino e teste
X = df['processed_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTamanho do conjunto de treino: {X_train.shape[0]}")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

# Vetorização do texto usando TF-IDF
print("\nVetorizando os textos usando TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Vetorização do texto usando Bag of Words
print("\nVetorizando os textos usando Bag of Words...")
count_vectorizer = CountVectorizer(max_features=5000)
X_train_bow = count_vectorizer.fit_transform(X_train)
X_test_bow = count_vectorizer.transform(X_test)


# Função para avaliar os modelos
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_type):
    # Treinamento do modelo
    print(f"\nTreinando o modelo {model_name} com {feature_type}...")
    model.fit(X_train, y_train)

    # Predição
    y_pred = model.predict(X_test)

    # Avaliação
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nResultados para {model_name} com {feature_type}:")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"\nRelátório de Classificação:\n{report}")

    # Plotagem da matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2], yticklabels=[1, 2])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name} with {feature_type}')
    plt.show()

    return accuracy, model


# Modelos
# 1. Naive Bayes
nb_model = MultinomialNB()

# 2. SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# 3. Random Forest (conforme solicitado)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Avaliação dos modelos com TF-IDF
print("\n" + "=" * 50)
print("AVALIAÇÃO DOS MODELOS COM TF-IDF")
print("=" * 50)

nb_accuracy_tfidf, nb_model_tfidf = evaluate_model(nb_model, X_train_tfidf, X_test_tfidf, y_train, y_test,
                                                   "Naive Bayes", "TF-IDF")
svm_accuracy_tfidf, svm_model_tfidf = evaluate_model(svm_model, X_train_tfidf, X_test_tfidf, y_train, y_test, "SVM",
                                                     "TF-IDF")
rf_accuracy_tfidf, rf_model_tfidf = evaluate_model(rf_model, X_train_tfidf, X_test_tfidf, y_train, y_test,
                                                   "Random Forest", "TF-IDF")

# Avaliação dos modelos com Bag of Words
print("\n" + "=" * 50)
print("AVALIAÇÃO DOS MODELOS COM BAG OF WORDS")
print("=" * 50)

nb_accuracy_bow, nb_model_bow = evaluate_model(MultinomialNB(), X_train_bow, X_test_bow, y_train, y_test, "Naive Bayes",
                                               "Bag of Words")
svm_accuracy_bow, svm_model_bow = evaluate_model(SVC(kernel='linear', C=1.0, random_state=42), X_train_bow, X_test_bow,
                                                 y_train, y_test, "SVM", "Bag of Words")
rf_accuracy_bow, rf_model_bow = evaluate_model(RandomForestClassifier(n_estimators=100, random_state=42), X_train_bow,
                                               X_test_bow, y_train, y_test, "Random Forest", "Bag of Words")

# Comparação dos resultados
results = {
    'TF-IDF': {
        'Naive Bayes': nb_accuracy_tfidf,
        'SVM': svm_accuracy_tfidf,
        'Random Forest': rf_accuracy_tfidf
    },
    'Bag of Words': {
        'Naive Bayes': nb_accuracy_bow,
        'SVM': svm_accuracy_bow,
        'Random Forest': rf_accuracy_bow
    }
}

# Plotagem dos resultados para comparação
models = ['Naive Bayes', 'SVM', 'Random Forest']
tfidf_scores = [results['TF-IDF'][model] for model in models]
bow_scores = [results['Bag of Words'][model] for model in models]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
tfidf_bars = ax.bar(x - width / 2, tfidf_scores, width, label='TF-IDF')
bow_bars = ax.bar(x + width / 2, bow_scores, width, label='Bag of Words')

ax.set_ylabel('Acurácia')
ax.set_title('Comparação dos Modelos por Tipo de Feature')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()


# Adicionar valores nas barras
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(tfidf_bars)
autolabel(bow_bars)

plt.tight_layout()
plt.show()

# Identificação do melhor modelo
best_accuracy = 0
best_model_name = ""
best_feature_type = ""

for feature_type in results:
    for model_name, accuracy in results[feature_type].items():
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_feature_type = feature_type

print("\n" + "=" * 50)
print(f"O melhor modelo é {best_model_name} com {best_feature_type}, alcançando uma acurácia de {best_accuracy:.4f}")
print("=" * 50)

# Resposta à pergunta do exercício
print("\nResposta à pergunta 1:")
print(
    "O bloco de código responsável por fazer o download dos módulos necessários do NLTK para a correta execução do programa consiste em:")
print("nltk.download('punkt')")
print("nltk.download('wordnet')")
print("nltk.download('averaged_perceptron_tagger')")
print("nltk.download('stopwords')")
print("\nA resposta correta é a alternativa a)")