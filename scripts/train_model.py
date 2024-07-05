from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
def train_random_forest(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    clf = RandomForestClassifier()
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_val)
    print("Random Forest Validation Score:", clf.score(X_val, y_val))
    print(classification_report(y_val, y_pred))
    dump(clf, '../models/rf_model.joblib')
    return clf

def train_gradient_boosting(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    clf = GradientBoostingClassifier()
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_val)
    print("Gradient Boosting Validation Score:", clf.score(X_val, y_val))
    print(classification_report(y_val, y_pred))
    dump(clf, '../models/gradient_boosting_model.joblib')
    return clf

def train_xgboost(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_val)
    print("XGBoost Validation Score:", clf.score(X_val, y_val))
    print(classification_report(y_val, y_pred))
    dump(clf, '../models/xgboost_model.joblib')
    return clf


def train_neural_network(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    param_grid = {
        'hidden_layer_sizes': [(100,), (100, 50), (100, 100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 300]
    }

    clf = MLPClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_res, y_train_res)

    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_val)
    print("Best parameters found:", grid_search.best_params_)
    print("Neural Network Validation Score:", best_clf.score(X_val, y_val))
    print(classification_report(y_val, y_pred))
    dump(best_clf, '../models/neural_network_model.joblib')
    return best_clf