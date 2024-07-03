import pandas as pd
import numpy as np
import joblib
import os
import shap
import dash # type: ignore
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from colorama import init, Fore, Style
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel

model = joblib.load('../results/model/xgb_model.joblib')

# Load the data, keeping SK_ID_CURR for output
X_train_full = pd.read_csv('../results/feature_engineering/feature_engineering_train.csv')
y_train = pd.read_csv('../data/application_train.csv')['TARGET']

X_test_full = pd.read_csv('../results/feature_engineering/feature_engineering_test.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train, test_size=0.2, random_state=42)

def color_print(color, message):
    color_code = getattr(Fore, color.upper(), Fore.WHITE)
    print(f"{color_code}{message}{Style.RESET_ALL}")

def print_step(message):
    color_print("cyan", message)

# Perform feature selection or load selected features
selected_features_file = '../data/selected_features.csv'

if os.path.exists(selected_features_file):
    print_step("Loading selected features from file...")
    selected_features = pd.read_csv(selected_features_file)['feature'].tolist()
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    color_print("green", f"Loaded selected features from file. Selected {X_train_selected.shape[1]} features.")
else:
    print_step("Performing feature selection...")
    selector = SelectFromModel(estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
    selector.fit(X_train, y_train.to_numpy().ravel())
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    selected_features = selector.get_feature_names_out()
    pd.DataFrame(selected_features, columns=['feature']).to_csv(selected_features_file, index=False)
    color_print("green", f"Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]}")
    color_print("green", "Selected features saved to file.")
    
            
# Load the selected features
selected_features_file = '../data/selected_features.csv'
selected_features = pd.read_csv(selected_features_file)['feature'].tolist()
if 'TARGET' in selected_features:
        # drop the target column
        selected_features.remove('TARGET')

# Dropping SK_ID_CURR for model input, but keep it for later reference
X_train = X_train_full[selected_features].to_numpy()
X_test = X_test_full[selected_features].to_numpy()

# Create the TreeExplainer
explainer = shap.TreeExplainer(model)

# Dash app setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Client Loan Decision Explanation"),
    dcc.Dropdown(
        id='client-id-dropdown',
        options=[{'label': str(sk_id), 'value': sk_id} for sk_id in X_test_full['SK_ID_CURR']],
        placeholder="Select a client ID"
    ),
    html.Div(id='output-container'),
    dcc.Graph(id='key-variables-bar')
])

@app.callback(
    [Output('output-container', 'children'),
     Output('key-variables-bar', 'figure')],
    [Input('client-id-dropdown', 'value')]
)
def update_output(selected_client_id):
    if selected_client_id is None:
        return "", {}

    # Find the index of the selected client ID
    client_index = X_test_full.index[X_test_full['SK_ID_CURR'] == selected_client_id].tolist()[0]

    # Generate SHAP force plot
    # Assuming X_test_full contains all necessary columns except 'TARGET', and selected_features now correctly excludes 'TARGET'
    selected_client_data = X_test_full[selected_features].iloc[client_index:client_index+1]
    shap_values = explainer.shap_values(selected_client_data)
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame(X_test, columns=selected_features).iloc[client_index])
    shap_html_path = f"../results/clients_outputs/force_plot_Test_{selected_client_id}.html"
    shap.save_html(shap_html_path, force_plot)
    
    # Generate key variables bar plot
    data_for_plot = X_test_full.iloc[client_index].drop('SK_ID_CURR').T.reset_index()
    data_for_plot.columns = ['Variable', 'Value']
    fig = px.bar(data_for_plot, x='Variable', y='Value', title=f'Key Variables for Client {selected_client_id}')
    
    # Load the true label
    true_label = y_test.iloc[client_index].values[0]
    
    # Predict and show result
    prediction = model.predict(X_test[client_index:client_index+1])[0]
    
    # Determine if the loan was given or not
    loan_given = 'Yes' if true_label == 1 else 'No'
    
    # Generate explanation text
    top_positive_features = np.argsort(shap_values[0])[-3:][::-1]
    top_negative_features = np.argsort(shap_values[0])[:3]

    explanation_text = f"Client SK_ID_CURR: {selected_client_id}, Prediction: {prediction}, Loan Given: {loan_given}. "
    explanation_text += "The most influential factors were: "

    for i in top_positive_features:
        explanation_text += f"{selected_features[i]} (positively), "
    
    for i in top_negative_features:
        explanation_text += f"{selected_features[i]} (negatively), "

    explanation_text = explanation_text.rstrip(", ") + "."

    return html.Div([
        html.Iframe(srcDoc=open(shap_html_path, 'r').read(), width='100%', height='400'),
        html.P(explanation_text)
    ]), fig

if __name__ == '__main__':
    app.run_server(debug=True)
