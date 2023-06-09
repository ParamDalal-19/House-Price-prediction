from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump, load


def train_model(model, X, y):
    model.fit(X, y)


def evaluate_model(model, X, y):
    scores = cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores


def save_model(model, file_path):
    dump(model, file_path)


def load_model(file_path):
    return load(file_path)


def predict_prices(model, X):
    return model.predict(X)


def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse
