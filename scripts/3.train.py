import pandas as pd
import xgboost as xgb
from joblib import dump
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

df_train = pd.read_csv('./results/feature_engineering/feature_engineering_train.csv')
app_train = pd.read_csv('./data/application_train.csv')

# Best parameters obtained from GridSearchCV
best_params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500}

# Instantiate the model with the best parameters
model = xgb.XGBClassifier(objective='binary:logistic', **best_params)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(df_train, app_train['TARGET'], test_size=0.2, random_state=42)

# Initialize lists to store training and validation scores
train_scores = []
val_scores = []

# Iterate through different sizes of training data with a progress bar
for i in tqdm(range(10, len(X_train), 50), desc="Training Progress"):
    # Create a smaller training set
    X_train_small = X_train[:i]
    y_train_small = y_train[:i]
    
    # Fit the model on the smaller training set
    model.fit(X_train_small, y_train_small)
    
    # Predict probabilities on the current training set and validation set
    xgb_train_pred = model.predict_proba(X_train_small)[:, 1]
    xgb_val_pred = model.predict_proba(X_val)[:, 1]
    
    # Append scores to the respective lists
    train_scores.append(roc_auc_score(y_train_small, xgb_train_pred))
    val_scores.append(roc_auc_score(y_val, xgb_val_pred))

# Plot the learning curves
plt.figure(figsize=(10, 6))
plt.plot(range(10, len(X_train), 50), train_scores, label='Training Score')
plt.plot(range(10, len(X_train), 50), val_scores, label='Validation Score')
plt.xlabel('Number of Training Samples')
plt.ylabel('ROC-AUC Score')
plt.title('Learning Curves with Best Parameters')
plt.legend()
plt.savefig('./results/learning_curves.png')
print("Learning curves plot saved successfully.")
plt.show()

dump(model, './results/model/xgb_model.joblib')
print("Best XGBoost model with parameters saved successfully.")

# Plot the feature importance of the model
col_names = df_train.columns
feat_imp = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': col_names, 'Importance': feat_imp})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(25, 10))
plt.barh(feat_imp_df['Feature'][:20], feat_imp_df['Importance'][:20])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 20 Features (XGBoost)')
plt.savefig('./results/feature_importance.png')
print("Feature importance plot saved successfully.")
plt.show()