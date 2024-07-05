from load_data import load_data
from train_model import train_random_forest, train_gradient_boosting, train_xgboost, train_neural_network
from test_model import test_model

# Load training data
X_train, y_train = load_data('../data/fraudTrain.csv')

# Train the models
print("Training Random Forest...")
train_random_forest(X_train, y_train)
print("Training Gradient Boosting...")
train_gradient_boosting(X_train, y_train)
print("Training XGBoost...")
train_xgboost(X_train, y_train)
print("Training Neural Network...")
train_neural_network(X_train, y_train)

# Load testing data
X_test, y_test = load_data('../data/fraudTest.csv')

# Test the models
print("Testing Random Forest...")
test_model(X_test, y_test, '../models/rf_model.joblib')
print("Testing Gradient Boosting...")
test_model(X_test, y_test, '../models/gradient_boosting_model.joblib')
print("Testing XGBoost...")
test_model(X_test, y_test, '../models/xgboost_model.joblib')
print("Testing Neural Network...")
test_model(X_test, y_test, '../models/neural_network_model.joblib')