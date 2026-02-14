import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/liver.csv")

# Convert gender to numeric
data['Gender'] = data['Gender'].map({'M':1,'F':0})

# Features and target
X = data.drop('Target', axis=1)
y = data['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, pred))
