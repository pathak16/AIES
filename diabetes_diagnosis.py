# diabetes_diagnosis.py

# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  # For Gradient Boosting (XGBoost)
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

# 4. Hyperparameter Tuning for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}

svm = SVC(random_state=42)
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, verbose=1, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)

best_svm_model = grid_search_svm.best_estimator_

# 5. Train the Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 6. Train the Gradient Boosting Model (XGBoost)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 7. Train the KNN Model (More Interpretable)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 8. Evaluate the Tuned SVM Model
print("\nEvaluating the Tuned SVM (Black Box) Model:")
y_pred_svm = best_svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

print("Tuned SVM Model Results:")
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

# 10. Evaluate the Gradient Boosting Model (XGBoost)
print("\nEvaluating the Gradient Boosting (XGBoost) Model:")
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
report_xgb = classification_report(y_test, y_pred_xgb)

print("Gradient Boosting (XGBoost) Model Results:")
print(f"Accuracy: {accuracy_xgb}")
print("Confusion Matrix:")
print(cm_xgb)
print("Classification Report:")
print(report_xgb)

# 11. Evaluate the KNN Model (More Transparent)
print("\nEvaluating the KNN (More Interpretable) Model:")
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
joblib.dump(best_svm_model, 'tuned_svm_diabetes_model.pkl')
joblib.dump(rf_model, 'random_forest_diabetes_model.pkl')
joblib.dump(xgb_model, 'xgb_diabetes_model.pkl')
joblib.dump(knn_model, 'knn_diabetes_model.pkl')
