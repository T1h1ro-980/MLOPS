from os import name
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib
import yaml


def load_config(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(config):
    df = pd.read_csv("data/scaled_df.csv")
    X = df.drop("quality", axis = 1)
    y = df["quality"]

    config_random_state = config["train"]["random_state"]
    config_test_size = config["train"]["test_size"]
    config_alpha = config["train"]["alpha"]
    config_l1_ratio = config["train"]["l1_ratio"]
    config_penalty = config["train"]["penalty"]
    config_loss = config["train"]["loss"]
    config_fit_intercept = config["train"]["fit_intercept"]


    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                    test_size=config_test_size,
                                                    random_state=config_random_state)
    

    params = {'alpha': config_alpha,
            'l1_ratio': config_l1_ratio,
            "penalty": config_penalty,
            "loss": config_loss,
            "fit_intercept": config_fit_intercept,
            }
    
    mlflow.set_experiment("linear model wine")
    with mlflow.start_run():
        lr = SGDRegressor(random_state=config_random_state)
        clf = GridSearchCV(lr, 
                            params, 
                            cv = 3, 
                            n_jobs = 4)
        clf.fit(X_train, y_train)
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        y_price_pred = y_pred
        (rmse, mae, r2)  = eval_metrics(y_val, y_price_pred)
        alpha = best.alpha
        l1_ratio = best.l1_ratio
        penalty = best.penalty
        eta0 = best.eta0
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("eta0", eta0)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("epsilon", best.epsilon)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        with open("best_wine.pkl", "wb") as file:
            joblib.dump(best, file)

if __name__ == "__main__":
    config = load_config("./src/params.yaml")
    train(config)