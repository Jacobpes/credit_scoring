import pandas as pd
import numpy as np
import joblib
import os
import shap
import plotly.express as px
from colorama import init, Fore, Style
from joblib import dump, load
from sklearn.model_selection import train_test_split

# Initialize colorama
init(autoreset=True)

def color_print(color, message):
    color_code = getattr(Fore, color.upper(), Fore.WHITE)
    print(f"{color_code}{message}{Style.RESET_ALL}")

def print_step(message):
    color_print("cyan", message)

# Function to make predictions
def make_predictions(model, data):
    print_step("Making predictions...")
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]  # if binary classification
    return predictions, probabilities

# Function to calculate and print the evaluation metrics
def evaluate_model(predictions, actual):
    accuracy = np.mean(predictions == actual)
    color_print("magenta", f"Accuracy: {accuracy:.2f}")

# Function to generate SHAP force plots and explanations
def generate_shap_force_plot_with_explanations(index, data, explainer, data_full, dataset_name, predictions, actuals):
    shap_values = explainer.shap_values(data.iloc[index:index+1])
    sk_id_curr = data_full.iloc[index]['SK_ID_CURR']
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], data.iloc[index])
    shap_html_path = f"../results/clients_outputs/force_plot_{dataset_name}_{sk_id_curr}.html"
    shap.save_html(shap_html_path, force_plot)
    
    top_positive_features = np.argsort(shap_values[0])[-3:][::-1]
    top_negative_features = np.argsort(shap_values[0])[:3]

    explanation_text = f"Client SK_ID_CURR: {sk_id_curr}, Prediction: {predictions[index]}, Actual: {actuals.iloc[index]}. "
    explanation_text += "The most influential factors were: "

    for i in top_positive_features:
        explanation_text += f"{data.columns[i]} (positively), "
    
    for i in top_negative_features:
        explanation_text += f"{data.columns[i]} (negatively), "

    explanation_text = explanation_text.rstrip(", ") + "."

    loan_status = "Approved" if predictions[index] == 1 else "Denied"
    color_print("green", f"Dataset: {dataset_name}, Client SK_ID_CURR: {sk_id_curr}, Prediction: {predictions[index]} ({loan_status}), Actual: {actuals.iloc[index]}")
    color_print("yellow", explanation_text)

    return shap_html_path, explanation_text

# Function for plotting key variables for a client
def plot_key_variables(index, data_full, dataset_name):
    data_for_plot = data_full.iloc[index].drop('SK_ID_CURR').T.reset_index()
    data_for_plot.columns = ['Variable', 'Value']
    fig = px.bar(data_for_plot, x='Variable', y='Value', title=f'Key Variables for {dataset_name} Client {index}')
    fig.show()

# Function for comparing two clients
def compare_clients(index1, index2, data_full):
    data1 = data_full.iloc[index1].drop('SK_ID_CURR').T
    data2 = data_full.iloc[index2].drop('SK_ID_CURR').T
    comparison_df = pd.DataFrame({'Client 1': data1, 'Client 2': data2})
    comparison_df.reset_index(inplace=True)
    fig = px.bar(comparison_df, x='index', y=['Client 1', 'Client 2'], barmode='group', title='Comparison between Two Training Clients')
    fig.write_image("../results/clients_outputs/comparison_plot.png")
    fig.show()

# Main script execution
if __name__ == "__main__":
    model = load('../results/model/xgb_model.joblib')
    color_print("green", "Model loaded successfully!")

    X_train_full = pd.read_csv('../results/feature_engineering/feature_engineering_train.csv')
    X_test_full = pd.read_csv('../results/feature_engineering/feature_engineering_test.csv')
    y_train = pd.read_csv('../data/application_train.csv')['TARGET']

    color_print("green", "Data loaded successfully!")

    # Load the selected features
    selected_features_file = '../data/selected_features.csv'
    print_step(f"Loading selected features from {selected_features_file}...")
    selected_features = pd.read_csv(selected_features_file)['feature'].tolist()
    print("Selected features:", selected_features)
    assert 'TARGET' not in selected_features, "'TARGET' should not be among selected features"
    color_print("green", "Selected features loaded successfully!")

    # Split the training data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_full[selected_features], y_train, test_size=0.2, random_state=42)

    # Convert to numpy arrays to avoid feature names warning
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    explainer = shap.TreeExplainer(model)

    train_predictions, _ = make_predictions(model, X_train)
    evaluate_model(train_predictions, y_train.to_numpy().ravel())

    test_predictions, _ = make_predictions(model, X_test)

    correct_indices = np.nonzero(train_predictions == y_train.to_numpy().ravel())[0]
    incorrect_indices = np.nonzero(train_predictions != y_train.to_numpy().ravel())[0]

    correct_example_index = correct_indices[0]
    incorrect_example_index = incorrect_indices[0]
    test_example_index = 0  # arbitrarily chosen, replace with the desired index

    shap_html_path_correct, explanation_correct = generate_shap_force_plot_with_explanations(correct_example_index, pd.DataFrame(X_train, columns=selected_features), explainer, X_train_full, 'Training', train_predictions, y_train)
    shap_html_path_incorrect, explanation_incorrect = generate_shap_force_plot_with_explanations(incorrect_example_index, pd.DataFrame(X_train, columns=selected_features), explainer, X_train_full, 'Training', train_predictions, y_train)
    shap_html_path_test, explanation_test = generate_shap_force_plot_with_explanations(test_example_index, pd.DataFrame(X_test, columns=selected_features), explainer, X_test_full, 'Test', test_predictions, y_test)

    plot_key_variables(correct_example_index, X_train_full, 'Training')
    plot_key_variables(incorrect_example_index, X_train_full, 'Training')
    plot_key_variables(test_example_index, X_test_full, 'Test')

    compare_clients(correct_example_index, incorrect_example_index, X_train_full)

    # Debug: Print some SK_ID_CURR values from the test dataset
    color_print("blue", "Some SK_ID_CURR values from the test dataset:")
    print(X_test_full['SK_ID_CURR'].head())

    # Generate HTML page with SHAP values and explanations
    html_content = f"""
    <html>
    <head>
        <title>Client Loan Decision Explanation</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .container {{ display: flex; flex-direction: column; align-items: center; }}
            .shap-plot, .explanation {{ width: 80%; margin: 20px 0; }}
            iframe {{ width: 100%; height: 400px; border: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Client Loan Decision Explanation</h1>
            
            <h2>Training Set</h2>
            <h3>Correct Example</h3>
            <div class="shap-plot"><iframe src="{shap_html_path_correct}"></iframe></div>
            <div class="explanation"><p>{explanation_correct}</p></div>
            
            <h3>Incorrect Example</h3>
            <div class="shap-plot"><iframe src="{shap_html_path_incorrect}"></iframe></div>
            <div class="explanation"><p>{explanation_incorrect}</p></div>
            
            <h2>Test Set</h2>
            <h3>Test Example</h3>
            <div class="shap-plot"><iframe src="{shap_html_path_test}"></iframe></div>
            <div class="explanation"><p>{explanation_test}</p></div>
        </div>
    </body>
    </html>
    """

    with open("../results/clients_outputs/client_loan_decision_explanation.html", "w") as file:
        file.write(html_content)

    color_print("magenta", "HTML page with SHAP values and explanations generated successfully!")

    # Continuously prompt for a valid client ID
    while True:
        specific_client_id = input("Enter the SK_ID_CURR of the client you want to analyze: ")
        client_indices = X_test_full.index[X_test_full['SK_ID_CURR'] == int(specific_client_id)].tolist()
        
        if not client_indices:
            color_print("red", f"Client with SK_ID_CURR {specific_client_id} not found in the test dataset.")
        else:
            specific_client_index = client_indices[0]
            
            # Generate SHAP force plot for the specific client
            shap_html_path_specific, explanation_specific = generate_shap_force_plot_with_explanations(specific_client_index, pd.DataFrame(X_test, columns=selected_features), explainer, X_test_full, 'Test', test_predictions, y_test)
            plot_key_variables(specific_client_index, X_test_full, 'Test')
            
            with open("../results/clients_outputs/client_loan_decision_explanation.html", "a") as file:
                file.write(f"""
                <div class="container">
                    <h3>Specific Client Example (SK_ID_CURR: {specific_client_id})</h3>
                    <div class="shap-plot"><iframe src="{shap_html_path_specific}"></iframe></div>
                    <div class="explanation"><p>{explanation_specific}</p></div>
                </div>
                """)
            
            break

    color_print("magenta", "Analysis completed!")
