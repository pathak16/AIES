# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load the Dataset
data = pd.read_csv('diabetes.csv')

# Preprocess the Data
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)  # Set random_state for consistency
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 4. Train the SVM Model
svm_model = SVC(kernel='rbf', C=1, gamma=1, random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred)
cm_svm = confusion_matrix(y_test, y_pred)
report_svm = classification_report(y_test, y_pred)


print("\nSVM Model Evaluation:")
print(f"Accuracy: {accuracy_svm:.16f}")
print("Confusion Matrix:")
print(cm_svm)
print("Classification Report:")
print(report_svm)

# 2. Train and Evaluate the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_split=2, min_samples_leaf=1, bootstrap=False, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy_rf:.16f}")
print("Confusion Matrix:")
print(cm_rf)
print("Classification Report:")
print(report_rf)

# 3. Train and Evaluate the Gradient Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42)

gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

accuracy_gb = accuracy_score(y_test, y_pred_gb)
cm_gb = confusion_matrix(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb)

print("\nGradient Boosting Model Evaluation:")
print(f"Accuracy: {accuracy_gb:.16f}")
print("Confusion Matrix:")
print(cm_gb)
print("Classification Report:")
print(report_gb)

# 4. Train and Evaluate the KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski')
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn)

print("\nKNN Model Evaluation:")
print(f"Accuracy: {accuracy_knn:.16f}")
print("Confusion Matrix:")
print(cm_knn)
print("Classification Report:")
print(report_knn)
