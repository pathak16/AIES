# diabetes_diagnosis2.py

# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load the Dataset
data = pd.read_csv('diabetes.csv')

# 3. Preprocess the Data
# Split features (X) and target (y)
X = data.drop('Outcome', axis=1)  # Features (all columns except 'Outcome')
y = data['Outcome']               # Target (Outcome column)

# 4. Split the Dataset into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=180, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Make Predictions on the Test Set
y_pred = rf_model.predict(X_test)

# 7. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Random Forest Accuracy: {accuracy}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)
