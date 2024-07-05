from sklearn.metrics import classification_report, accuracy_score
from joblib import load

def test_model(X_test, y_test, model_path):
    # Load the model
    clf = load(model_path)
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))