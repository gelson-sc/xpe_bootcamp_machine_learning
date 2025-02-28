from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

df = pd.read_csv('diabetes_numeric.csv')
df.head()
X = df.drop(columns=["c_peptide"])  # Usar 'age' e 'deficit' como preditores
y = df["c_peptide"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.37, random_state=5762)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(r2)
