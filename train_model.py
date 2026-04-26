import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# # Load dataset
# # CSV columns: TV, Radio, Newspaper, Sales
    
df = pd.read_csv(r'C:\Users\haris\Downloads\Advertising.csv')
df.head()

X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print('R2 Score:', r2_score(y_test, pred))
print('MSE:', mean_squared_error(y_test, pred))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model saved as model.pkl')