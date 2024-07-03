# Implement a program that takes as input the trained model, the customer id ... and returns:

# the score and the SHAP force plot associated with it
# Plotly visualization that show:
# key variables describing the client and its loan(s)
# comparison between this client and other clients
# Choose the 3 clients of your choice, compute the score, run the visualizations on their data and save them.

# Take 2 clients from the train set:
# 1 on which the model is correct and the other on which the model is wrong. Try to understand why the model got wrong on this client.
# Take 1 client from the test set

import json
import pandas as pd
import shap
import plotly.express as px
from joblib import load

def analyze_client(model, data, client_id):
    # Extract the client's data
    client_data = data[data['SK_ID_CURR'] == client_id]

    # Predict the score
    score = model.predict_proba(client_data.drop('SK_ID_CURR', axis=1))[:,1]

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(client_data.drop('SK_ID_CURR', axis=1))

    # SHAP force plot
    shap.force_plot(explainer.expected_value, shap_values, client_data.drop('SK_ID_CURR', axis=1))

    # Plotly visualization: key variables and comparison
    fig = px.bar(client_data, x=client_data.columns, y=client_data.values.squeeze())
    fig.show()

    return score, fig

# Load the model and data
model = load('./results/model/xgb_model.joblib')
df_test = pd.read_csv('./results/feature_engineering/feature_engineering_test.csv')
df_train = pd.read_csv('./results/feature_engineering/feature_engineering_train.csv')
data_df = pd.concat([df_train, df_test])

# Example usage for a specific client
client_id = 1  # Adjust this ID based on your dataset
score, fig = analyze_client(model, data_df, client_id)

# Save the scores and explanation in a JSON file for further inspection
results = {'client_id': client_id, 'score': score}
with open(f'results_{client_id}.json', 'w') as f:
    json.dump(results, f)

# To save Plotly figures:
fig.write_image(f'client_{client_id}_plot.png')

# predict on the train set to get the id of one successful prediction and one failed prediction
train = pd.read_csv('./data/application_train.csv')
train['SK_ID_CURR'] = train['SK_ID_CURR'].astype(int)