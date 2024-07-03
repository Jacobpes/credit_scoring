import pandas as pd
from joblib import dump, load
# train test split
from sklearn.model_selection import train_test_split
# roc_auc_score
from sklearn.metrics import roc_auc_score

# Load the feature-engineered data
df_train = pd.read_csv('./results/feature_engineering/feature_engineering_train.csv')
df_test = pd.read_csv('./results/feature_engineering/feature_engineering_test.csv')
app_test = pd.read_csv('./data/application_test.csv')
app_train = pd.read_csv('./data/application_train.csv')

# Load the model using joblib.load
model = load('./results/model/xgb_model.joblib')

# Predictions for kaggle submission
xgb_pred_test = model.predict_proba(df_test)[:,1]

# Prepare the submission DataFrame
submit = pd.DataFrame({'SK_ID_CURR': app_test['SK_ID_CURR'], 'TARGET': xgb_pred_test})

# Save the submission CSV
submit.to_csv('xgb_experiment.csv', index=False)

print("XGBoost submission saved to xgb_experiment.csv successfully.")