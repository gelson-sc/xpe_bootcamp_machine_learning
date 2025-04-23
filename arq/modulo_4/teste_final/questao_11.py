import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('fall_detection.csv')
X = df[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']]
y = df['ACTIVITY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rfc = RandomForestClassifier(n_estimators=50, random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
