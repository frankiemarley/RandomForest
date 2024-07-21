import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

# Load and preprocess data
data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')

# Handle zero values
zero_columns = ['Glucose', 'BloodPressure', 'BMI', 'Insulin']
for column in zero_columns:
    data[column] = data[column].replace(0, np.nan)

# Impute missing values
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Feature engineering
data_imputed['BMI_Category'] = pd.cut(data_imputed['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
data_imputed['Age_Group'] = pd.cut(data_imputed['Age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])

# Prepare data for modeling
X = data_imputed.drop(columns=['Outcome', 'BMI_Category', 'Age_Group'])
y = data_imputed['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Address class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature selection
rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
X_train_selected = rfe.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = rfe.transform(X_test)

# Define models
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

# Simplified hyperparameter grids
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform Grid Search with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use RandomizedSearchCV with fewer iterations
rf_random = RandomizedSearchCV(rf_model, rf_params, n_iter=20, cv=5, n_jobs=-1, random_state=42)
rf_random.fit(X_train_selected, y_train_resampled)

gb_random = RandomizedSearchCV(gb_model, gb_params, n_iter=20, cv=5, n_jobs=-1, random_state=42)
gb_random.fit(X_train_selected, y_train_resampled)

# Use the best models for prediction and evaluation
best_rf = rf_random.best_estimator_
best_gb = gb_random.best_estimator_

# Evaluate models
for name, model in [('Random Forest', best_rf), ('Gradient Boosting', best_gb)]:
    y_pred = model.predict(X_test_selected)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Feature importance (for Random Forest)
feature_importances = pd.Series(best_rf.feature_importances_, index=rfe.get_feature_names_out())
plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=True).plot(kind='barh')
plt.title('Feature Importances (Random Forest)')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'], 
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(y_test, best_rf.predict(X_test_selected), 'Random Forest Confusion Matrix')
plot_confusion_matrix(y_test, best_gb.predict(X_test_selected), 'Gradient Boosting Confusion Matrix')

# Feature Importance
def plot_feature_importance(model, feature_names, title):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    plt.figure(figsize=(10, 6))
    importances.sort_values(ascending=True).plot(kind='barh')
    plt.title(title)
    plt.xlabel('Relative Importance')
    plt.show()

plot_feature_importance(best_rf, rfe.get_feature_names_out(), 'Random Forest Feature Importance')
plot_feature_importance(best_gb, rfe.get_feature_names_out(), 'Gradient Boosting Feature Importance')

# Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Distribution of features
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
axes = axes.ravel()

for idx, col in enumerate(data.columns):
    sns.histplot(data=data, x=col, hue='Outcome', kde=True, ax=axes[idx])
    axes[idx].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(data, hue='Outcome', vars=['Glucose', 'BMI', 'Age', 'Insulin'], height=3)
plt.suptitle('Pairplot of Key Features', y=1.02)
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(model, X_test, y_test, model_name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve(best_rf, X_test_selected, y_test, 'Random Forest')
plot_roc_curve(best_gb, X_test_selected, y_test, 'Gradient Boosting')