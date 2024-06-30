import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump

# Initialize the GradientBoostingClassifier
model_gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Load the datasets
df_train = pd.read_csv('./results/feature_engineering.csv')
df_test = pd.read_csv('./results/feature_engineering_test.csv')
app_train = pd.read_csv('./data/application_train.csv')
app_test = pd.read_csv('./data/application_test.csv')

# Train the model on the training data
model_gb.fit(df_train, app_train['TARGET'])

# Predict probabilities on the training set
gb_train_pred = model_gb.predict_proba(df_train)[:, 1]
roc_score_gb = roc_auc_score(app_train['TARGET'], gb_train_pred)

print(f"Gradient Boosting ROC-AUC score on training data: {roc_score_gb}")

# Get the feature importance of the model to plot it
col_names = df_train.columns
feat_imp_gb = model_gb.feature_importances_
feat_imp_df_gb = pd.DataFrame({'Feature': col_names, 'Importance': feat_imp_gb})
feat_imp_df_gb = feat_imp_df_gb.sort_values(by='Importance', ascending=False)

# Save the model using joblib.dump
dump(model_gb, './results/gb_model.joblib')

print("Gradient Boosting model saved successfully.")

# Plot the importance of the top 20 features
plt.figure(figsize=(25, 10))
plt.barh(feat_imp_df_gb['Feature'][:20], feat_imp_df_gb['Importance'][:20])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 20 Features (Gradient Boosting)')
plt.show()
