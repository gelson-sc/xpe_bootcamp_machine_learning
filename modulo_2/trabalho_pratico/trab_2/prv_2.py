import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
df = pd.read_csv(url, header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
print(df.head(3))
print(df.describe())
# print(df.info())
# print(df.shape)
# print(df.isnull().sum())
# mediana = df["variance"].median()
# print(mediana)
# desvio_padrao = df["curtosis"].std()
# print(desvio_padrao)
# porcentagem_falsas = (df["class"].value_counts(normalize=True)[1]) * 100
# print(porcentagem_falsas)
# Calcular a correlação de Pearson entre "skewness" e "curtosis"
correlacao_pearson = df["skewness"].corr(df["curtosis"], method="pearson")

print(correlacao_pearson)