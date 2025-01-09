from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the model and scaler
model = joblib.load('diabetes_classifier.pkl')
scaler = joblib.load('scaler.pkl')

# Load the dataset (assuming it's available in the same directory)
data = pd.read_csv('diabetes_dataset.csv')

# Preprocess the data
X = data.drop('type', axis=1)  # Features
y = data['type']  # Labels

# Normalize the features using the same scaler used for training
X_scaled = scaler.transform(X)

# Make predictions on the entire dataset
y_pred = model.predict(X_scaled)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
classification_rep = classification_report(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', accuracy=accuracy, classification_rep=classification_rep, conf_matrix=conf_matrix)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        glucose = float(request.form['glucose'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = float(request.form['age'])
        skin_thickness = float(request.form['skin_thickness'])

        patient_data = np.array([[glucose, insulin, bmi, age, skin_thickness]])
        patient_data = scaler.transform(patient_data)

        prediction = model.predict(patient_data)[0]
        if prediction == 0:
            result = "Type 1 Diabetes"
        elif prediction == 1:
            result = "Type 2 Diabetes"
        else:
            result = "Healthy"

        # Generate graphs
        generate_graphs(glucose, insulin, bmi, age, skin_thickness, result)

        return render_template('result.html', prediction_text=result)
    except ValueError:
        return render_template('result.html', prediction_text="Error: Invalid input data. Please enter valid numbers.")

@app.route('/graphs/<filename>')
def get_graph(filename):
    return send_from_directory('graphs', filename)

def generate_graphs(glucose, insulin, bmi, age, skin_thickness, result):
    # Ensure the 'graphs' directory exists
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    features = ['Glucose', 'Insulin', 'BMI', 'Age', 'Skin Thickness']
    values = [glucose, insulin, bmi, age, skin_thickness]

    # Bar chart
    plt.figure()
    plt.bar(features, values, color='blue')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title(f'Patient Data - {result}')
    plt.savefig('graphs/bar_chart.png')
    plt.close()

    # Pie chart
    plt.figure()
    plt.pie(values, labels=features, autopct='%1.1f%%', startangle=140)
    plt.title(f'Feature Distribution - {result}')
    plt.savefig('graphs/pie_chart.png')
    plt.close()

    # Line chart
    plt.figure()
    plt.plot(features, values, marker='o')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title(f'Patient Data Trend - {result}')
    plt.savefig('graphs/line_chart.png')
    plt.close()

if __name__ == "__main__":
    app.run(debug=True)





