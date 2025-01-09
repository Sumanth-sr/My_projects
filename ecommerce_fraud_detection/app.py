import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import joblib
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# In-memory user storage (use a database in a real application)
users = {}

# Function to load dataset
def load_data():
    from sklearn.datasets import make_classification
    data, labels = make_classification(
        n_samples=5000, n_features=20, n_informative=10, n_redundant=5,
        n_clusters_per_class=1, weights=[0.99], flip_y=0.01, random_state=42
    )
    return pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(20)]), labels

data, labels = load_data()
data['Label'] = labels
data.to_csv('synthetic_fraud_dataset.csv', index=False)
print("Synthetic dataset saved as 'synthetic_fraud_dataset.csv'.")

# Step 2: Split the dataset
X = data.drop(columns=['Label'])
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train a supervised learning model (Random Forest)
rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Step 4: Train an unsupervised learning model (Isolation Forest)
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
isolation_forest.fit(X_train)

# Step 5: Combine predictions
supervised_preds = rf.predict_proba(X_test)[:, 1]
anomaly_scores = isolation_forest.decision_function(X_test)
combined_scores = 0.7 * supervised_preds + 0.3 * (1 - anomaly_scores)

threshold = 0.5
final_predictions = (combined_scores > threshold).astype(int)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['GET'])
def evaluate():
    evaluation_results = {
        "accuracy": accuracy_score(y_test, final_predictions),
        "confusion_matrix": confusion_matrix(y_test, final_predictions).tolist(),
        "classification_report": classification_report(y_test, final_predictions, output_dict=True)
    }
    return jsonify(evaluation_results)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    amount = float(data['amount'])
    payment_method = data['payment_method']
    product_category = data['product_category']
    quantity = float(data['quantity'])
    customer_age = float(data['customer_age'])
    device = data['device']
    account_age_days = float(data['account_age_days'])
    transaction_hour = float(data['transaction_hour'])
    address_match = data['address_match'] == "Yes"
    model_type = data['model']

    feature_vector = np.zeros(20)
    feature_vector[0] = amount
    feature_vector[1] = quantity
    feature_vector[2] = customer_age
    feature_vector[3] = account_age_days
    feature_vector[4] = transaction_hour
    feature_vector[5] = 1 if payment_method == 'Credit Card' else 0
    feature_vector[6] = 1 if payment_method == 'Debit Card' else 0
    feature_vector[7] = 1 if payment_method == 'UPI' else 0
    feature_vector[8] = 1 if payment_method == 'Net Banking' else 0
    feature_vector[9] = 1 if address_match else 0

    features = scaler.transform([feature_vector])
    
    if model_type == 'Random Forest':
        model = rf
    elif model_type == 'Isolation Forest':
        model = isolation_forest
    else:
        supervised_pred = rf.predict_proba(features)[:, 1]
        anomaly_score = isolation_forest.decision_function(features)
        combined_score = 0.7 * supervised_pred + 0.3 * (1 - anomaly_score)
        prediction = "Fraud" if combined_score > threshold else "Not Fraud"
        return render_template('result.html', prediction=prediction, combined_score=combined_score[0])

    if model_type == 'Random Forest':
        supervised_pred = model.predict_proba(features)[:, 1]
        prediction = "Fraud" if supervised_pred > threshold else "Not Fraud"
    else:
        anomaly_score = model.decision_function(features)
        prediction = "Fraud" if anomaly_score < 0 else "Not Fraud"

    return render_template('result.html', prediction=prediction, combined_score=supervised_pred[0] if model_type == 'Random Forest' else anomaly_score[0])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists', 'error')
        else:
            hashed_password = generate_password_hash(password)
            user = User(id=len(users) + 1, username=username, password=hashed_password)
            users[username] = user
            flash('Account created successfully. Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == int(user_id):
            return user
    return None

if __name__ == '__main__':
    app.run(debug=True)
