import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import xgboost as xgb 
from joblib import dump

model = xgb.XGBClassifier(
    n_estimators = 1000,  
    max_depth=3,       
    learning_rate=0.1, 
    objective='binary:logistic'  
)

df_train = pd.read_csv('./results/feature_engineering.csv')
df_test = pd.read_csv('./results/feature_engineering_test.csv')
app_train = pd.read_csv('./data/application_train.csv')
app_test = pd.read_csv('./data/application_test.csv')

# Train the model on the training data
model.fit(df_train, app_train['TARGET'])

xgb_train_pred = model.predict_proba(df_train)[:, 1]  
roc_score_xgb = roc_auc_score(app_train['TARGET'], xgb_train_pred)

# get the feature importance of the model to plot
col_names = df_train.columns
feat_imp = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': col_names, 'Importance': feat_imp})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Save the model using joblib.dump
dump(model, './results/xgb_model.joblib')

print("XGBoost model saved to ./results/xgb_model.joblib successfully.")

# plot the importance of the top 20 features
plt.figure(figsize=(25, 10))
plt.barh(feat_imp_df['Feature'][:20], feat_imp_df['Importance'][:20])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Features')
plt.show()


