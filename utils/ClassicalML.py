import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from utils.GLOBALS import device
from utils.new_preproc import inverse_min_max_scaling


def train_rf(X_train, y_train, X_test, y_test, n_estimators=10):
    """
    Trains a Random Forest model
    :param X_train: train data
    :param y_train: train labels
    :param X_test: test data
    :param y_test: test labels
    :param n_estimators: hyperparameters for rf
    :return: None
    """
    random_forest_model = RandomForestRegressor(n_estimators=n_estimators)
    random_forest_model.fit(X_train, y_train)

    y_train_pred = random_forest_model.predict(X_train)
    mae = mean_absolute_error(y_train, y_train_pred)
    mse = mean_squared_error(y_train, y_train_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse)}")

    y_pred = random_forest_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse)}")


def train_gbr(X_train, y_train, X_test, y_test, n_estimators=10):
    """
    Trains a Gradient Boosting model
    :param X_train: train data
    :param y_train: train labels
    :param X_test: test data
    :param y_test: test labels
    :param n_estimators: hyperparameters for gbr
    :return: None
    """
    gbr_model = GradientBoostingRegressor(n_estimators=n_estimators)
    gbr_model.fit(X_train, y_train)

    y_train_pred = gbr_model.predict(X_train)
    mae = mean_absolute_error(y_train, y_train_pred)
    mse = mean_squared_error(y_train, y_train_pred)

    print(f"Train Mean Absolute Error (MAE): {mae}")
    print(f"Train Root Mean Squared Error (RMSE): {np.sqrt(mse)}")

    y_pred = gbr_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Test Mean Absolute Error (MAE): {mae}")
    print(f"Test Root Mean Squared Error (RMSE): {np.sqrt(mse)}")


def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Trains a Linear Regression model
    :param X_train: train data
    :param y_train: train labels
    :param X_test: test data
    :param y_test: test labels
    :return: None
    """
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)

    y_train_pred = linear_reg.predict(X_train)
    mae = mean_absolute_error(y_train, y_train_pred)
    mse = mean_squared_error(y_train, y_train_pred)

    print(f"Train Mean Absolute Error (MAE): {mae}")
    print(f"Train Root Mean Squared Error (RMSE): {np.sqrt(mse)}")

    y_pred = linear_reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Test Mean Absolute Error (MAE): {mae}")
    print(f"Test Root Mean Squared Error (RMSE): {np.sqrt(mse)}")


def train_xgboost(X_train, y_train, X_test, y_test, n_estimators=10, learning_rate=0.09, max_depth=10):
    """
    Trains a XGBoost model
    :param X_train: train data
    :param y_train: train labels
    :param X_test: test data
    :param y_test: test labels
    :param n_estimators: hyperparameters for xgb
    :param learning_rate: hyperparameters for xgb
    :param max_depth: hyperparameters for xgb
    :return: None
    """
    xgboost_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    xgboost_model.fit(X_train, y_train)

    y_train_pred = xgboost_model.predict(X_train)
    mae = mean_absolute_error(y_train, y_train_pred)
    mse = mean_squared_error(y_train, y_train_pred)

    print(f"Train Mean Absolute Error (MAE): {mae}")
    print(f"Train Root Mean Squared Error (RMSE): {np.sqrt(mse)}")

    y_pred = xgboost_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Test Mean Absolute Error (MAE): {mae}")
    print(f"Test Root Mean Squared Error (RMSE): {np.sqrt(mse)}")


def get_classical_ml_train_test_data(model, all_train_data_loader, test_data_loader):
    features_list = []
    targets_list = []
    model.eval()
    with torch.no_grad():
        for search_term, product_description, relevance in all_train_data_loader:
            search_term, product_description, relevance = search_term.to(device), product_description.to(
                device), relevance.to(device)

            outputs_search, outputs_description = model.get_outputs(search_term, product_description)
            features = torch.cat((outputs_search, outputs_description), dim=1)

            features_list.append(features.cpu().numpy())
            targets_list.append(inverse_min_max_scaling(relevance.cpu().numpy()))

    X_train = np.concatenate(features_list, axis=0)
    y_train = np.concatenate(targets_list, axis=0)

    features_list = []
    targets_list = []
    model.eval()
    with torch.no_grad():
        for search_term, product_description, relevance in test_data_loader:
            search_term, product_description, relevance = search_term.to(device), product_description.to(
                device), relevance.to(device)

            outputs_search, outputs_description = model.get_outputs(search_term, product_description)
            features = torch.cat((outputs_search, outputs_description), dim=1)

            features_list.append(features.cpu().numpy())
            targets_list.append(inverse_min_max_scaling(relevance.cpu().numpy()))

    X_test = np.concatenate(features_list, axis=0)
    y_test = np.concatenate(targets_list, axis=0)
    return X_train, y_train, X_test, y_test
