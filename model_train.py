import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklego.mixture import GMMOutlierDetector

from logger import logger


def train_test_split(dataf: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split the data into train and test sets

    Parameters
    ----------
    data : polars.DataFrame
        The data to split

    Returns
    -------
    tuple[polars.DataFrame, polars.DataFrame]
        The train and test sets
    """
    logger.info("Splitting data into train and test sets")
    X_train = dataf.filter(pl.col("date") < pl.datetime(2024, 4, 1))

    X_test = dataf.filter(pl.col("date") >= pl.datetime(2024, 4, 1))
    return X_train, X_test


# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2


def get_params(model):
    n_estimators = model.get_params()["n_estimators"]
    return n_estimators


def train_model(data: pl.DataFrame, n_components: int = 4*7, threshold: float = 0.99) -> None:
    """
    Train the model
    """
    X_train, X_test = train_test_split(data)
    
    with mlflow.start_run():
        if os.path.exists("models/mod_pipe.joblib"):
            mod_pipe = joblib.load("models/mod_pipe.joblib")
        else:
            feat_pipe = make_column_transformer(
                (OneHotEncoder(sparse_output=False), ["country"]),
                (OrdinalEncoder(categories=[["Y", "X", "W", "T"]]), ["os"]),
                (StandardScaler(), ["count"])
            )
            mod_pipe = make_pipeline(
                feat_pipe,
                GMMOutlierDetector(n_components=n_components, threshold=threshold)
            )
        mod_pipe.fit(X_train)
        anom_labels = mod_pipe.predict(X_test)

        # n_estimators = get_params(model)

        # rmse, mae, r2 = eval_metrics(X_test, predicted_qualities)

        # logger.info(f"n_estimators: {n_estimators}")

        # mlflow.log_param("n_estimators", n_estimators)

        df = None
        if os.path.exists("data/processed/ad_baseline.csv"):
            df = pl.read_csv("data/processed/ad_baseline.csv", try_parse_dates=True)
            df_baseline = df.clone()
        else:
            df_baseline = X_train
        # drift = eval_drift(X_train, df_baseline, model)
        
        # logger.info(f"RMSE: {rmse}")
        # logger.info(f"MAE: {mae}")
        # logger.info(f"R2: {r2}")
        # logger.info(f"AUC: {drift}")

        # mlflow.log_metric("rmse", rmse)
        # mlflow.log_metric("r2", r2)
        # mlflow.log_metric("mae", mae)
        # mlflow.log_metric("auc", drift)

        mlflow.sklearn.log_model(mod_pipe, "model_pipeline")
        joblib.dump(mod_pipe, "models/mod_pipe.joblib")
        if df is None:
            df = data
        else:
            df = pl.concat([df, data])
        df.write_csv("data/processed/ad_baseline.csv")


if __name__ == "__main__":
    data = pl.read_csv("data/processed/ad_preprocessed.csv", try_parse_dates=True)
    train_model(data)