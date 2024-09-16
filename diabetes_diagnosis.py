# diabetes_diagnosis.py

# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 2. Load the Dataset
data = pd.read_csv('diabetes.csv')

# 3. Preprocess the Data
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train the SVM Model
svm_model = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
svm_model.fit(X_train, y_train)

# 5. Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Train the Gradient Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# 7. Train the KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 8. Evaluate the SVM Model
print("\nEvaluating the SVM Model:")
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

print("SVM Model Results:")
print(f"Accuracy: {accuracy_svm}")
print("Confusion Matrix:")
print(cm_svm)
print("Classification Report:")
print(report_svm)

# 9. Evaluate the Random Forest Model
print("\nEvaluating the Random Forest Model:")
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print("Random Forest Model Results:")
print(f"Accuracy: {accuracy_rf}")
print("Confusion Matrix:")
print(cm_rf)
print("Classification Report:")
print(report_rf)

# 10. Evaluate the Gradient Boosting Model
print("\nEvaluating the Gradient Boosting Model:")
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
cm_gb = confusion_matrix(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb)

print("Gradient Boosting Model Results:")
print(f"Accuracy: {accuracy_gb}")
print("Confusion Matrix:")
print(cm_gb)
print("Classification Report:")
print(report_gb)

# 11. Evaluate the KNN Model
print("\nEvaluating the KNN Model:")
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn)

print("KNN Model Results:")
print(f"Accuracy: {accuracy_knn}")
print("Confusion Matrix:")
print(cm_knn)
print("Classification Report:")
print(report_knn)

# 12. Save the Models
joblib.dump(svm_model, 'svm_diabetes_model.pkl')
joblib.dump(rf_model, 'random_forest_diabetes_model.pkl')
joblib.dump(gb_model, 'gb_diabetes_model.pkl')
joblib.dump(knn_model, 'knn_diabetes_model.pkl')
