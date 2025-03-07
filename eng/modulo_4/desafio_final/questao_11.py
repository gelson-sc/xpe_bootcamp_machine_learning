import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('cars_validade.csv')
hp = df[['hp']]
scaler = StandardScaler()
hp_scaled = scaler.fit_transform(hp)
print(hp_scaled)
max_hp_scaled = hp_scaled.max()
print(max_hp_scaled)