import pandas as pd
from joblib import load

# Load the feature-engineered data
df_test = pd.read_csv('./results/feature_engineering/feature_engineering_test.csv')
app_test = pd.read_csv('./data/application_test.csv')

# Load the xgb model using joblib.load
xgb_model = load('./results/model/xgb_model.joblib')
# Load the rf model using joblib.load
rf_model = load('./results/model/rf_model.joblib')

# Predictions
xgb_pred_test = xgb_model.predict_proba(df_test)[:,1]
rf_pred_test = rf_model.predict_proba(df_test)[:,1]

# Voting
xgb_weight = 0.7
rf_weight = 0.3
vote_soft_pred = xgb_weight * xgb_pred_test + rf_weight * rf_pred_test

# Prepare the submission DataFrame
submit = pd.DataFrame({'SK_ID_CURR': app_test['SK_ID_CURR'], 'TARGET': vote_soft_pred})

# Save the submission CSV
submit.to_csv('xgb_rf_voting_soft.csv', index=False)

print("XGBoost submission saved to xgb_experiment.csv successfully.")
