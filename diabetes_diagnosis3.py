# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load the Dataset
data = pd.read_csv('diabetes.csv')  # Replace with your dataset path

# 3. Preprocess the Data
X = data.drop('Outcome', axis=1)  # Assuming 'Outcome' is the target variable
y = data['Outcome']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train the Gradient Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=140, learning_rate=0.11 , max_depth=4, min_samples_split=4, min_samples_leaf=4, random_state=42)
gb_model.fit(X_train, y_train)

# 5. Make Predictions and Evaluate the Model
y_pred = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Gradient Boosting Model Accuracy: {accuracy}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)
