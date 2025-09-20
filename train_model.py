import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_logistic_regression(X, y, param_grid=None, cv=3):
    if param_grid is None:
        param_grid = {
            'estimator__C': [0.1, 1, 10],
            'estimator__penalty': ['l2'],
            'estimator__solver': ['lbfgs']
        }
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, scoring='f1_micro', verbose=2, n_jobs=-1)
    grid.fit(X, y)
    print("Best params:", grid.best_params_)
    best_model = grid.best_estimator_
    return best_model

def evaluate_model(model, X_test, y_test, mlb):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

if __name__ == "__main__":
    from sklearn.preprocessing import MultiLabelBinarizer
    X_dummy = np.random.rand(100, 300)
    y_dummy = [['410.9', '786.05'], ['250.00'], ['346.0'], ['496'], ['540.9']] * 20
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y_dummy)
    model = train_logistic_regression(X_dummy, y_bin)
    evaluate_model(model, X_dummy, y_bin, mlb)
