import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump

# Initialize the RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42)

# Load the datasets
df_train = pd.read_csv('./results/feature_engineering/feature_engineering_train.csv')
app_train = pd.read_csv('./data/application_train.csv')

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [1000, 2000],
    'max_depth': [2, 3],
}

# Create an instance of GridSearchCV
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=2, scoring='roc_auc_ovr', verbose=2)

# Fit the GridSearchCV object to the training data
grid_search.fit(df_train, app_train['TARGET'])

# Retrieve the best estimator
best_model = grid_search.best_estimator_

# Predict probabilities on the training set using the best model
rf_train_pred_best = best_model.predict_proba(df_train)[:, 1]
roc_score_rf_best = roc_auc_score(app_train['TARGET'], rf_train_pred_best)

print(f"Best RandomForest ROC-AUC score on training data: {roc_score_rf_best}")
print(f"Best parameters: {grid_search.best_params_}")
# Save the best model using joblib.dump
dump(best_model, './results/model/rf_model.joblib')

print("Best RandomForest model saved successfully.")
