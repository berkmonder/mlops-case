import os

import joblib
import polars as pl
import mlflow
from mlflow.models import infer_signature
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklego.mixture import GMMOutlierDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    return accuracy, precision, recall, f1

def get_params(model_pipeline) -> tuple[int, float]:
    n_components = model_pipeline.get_params()["gmmoutlierdetector"].n_components
    threshold = model_pipeline.get_params()["gmmoutlierdetector"].threshold
    return n_components, threshold

def train_model(data: pl.DataFrame, n_components: int = 4*7, threshold: float = 0.99) -> None:
    """
    Train the model and save it to disk

    Parameters
    ----------
    data : polars.DataFrame
        The data to train the model on
    n_components : int
        The number of components
    threshold : float
        The threshold
    """
    X_train, X_test = train_test_split(data)

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
    # anom_labels = mod_pipe.predict(X_test)

    self_labels = [
        ["null", "XX", "W", 553033, -1],
        ["null", "XX", "X", 1435442, 1],
        ["null", "XX", "X", 1474590, 1],
        ["null", "XX", "Y", 4534055, 1],
        ["null", "XX", "Y", 4207723, 1],
        ["null", "XZ", "X", 249004, 1],
        ["null", "XZ", "X", 254206, 1],
        ["null", "XZ", "W", 16917, 1],
        ["null", "XZ", "W", 15238, 1],
        ["null", "XZ", "Y", 1137806, 1],
        ["null", "XZ", "Y", 1122380, 1],
        ["null", "YX", "W", 13363, 1],
        ["null", "YX", "W", 12632, 1],
        ["null", "YX", "X", 89531, 1],
        ["null", "YX", "X", 79945, 1],
        ["null", "YX", "Y", 371325, 1],
        ["null", "YX", "Y", 352105, 1],
        ["null", "YY", "W", 11051, 1],
        ["null", "YY", "W", 9485, 1],
        ["null", "YY", "W", 7026, -1],
        ["null", "YY", "X", 375756, -1],
        ["null", "YY", "X", 438198, 1],
        ["null", "YY", "X", 440438, 1],
        ["null", "YY", "Y", 2892479, -1],
        ["null", "YY", "Y", 4033516, 1],
        ["null", "YY", "Y", 3866394, 1],
        ["null", "ZX", "Y", 149602, 1],
        ["null", "ZX", "Y", 139554, 1],]

    schema = ["date", "country", "os", "count", "label"]

    data = pl.DataFrame(data=self_labels, schema=schema)

    y_sample_true = data.get_column("label")
    X_test_sample = data.select(["date", "country", "os", "count"])

    y_pred = mod_pipe.predict(X_test_sample)
    accuracy, precision, recall, f1 = eval_metrics(y_sample_true, y_pred)

    # Track model with MLflow
    mlflow.set_tracking_uri(uri="http://mlflow:5000")
    
    mlflow.set_experiment("MLOps Case Study - Anomaly Detection")

    with mlflow.start_run():
        n_components, threshold = get_params(mod_pipe)
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("threshold", threshold)

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f1}")
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.set_tag("Training info", "Anomaly Detection with GMMOutlierDetector")

        signature = infer_signature(X_train.to_pandas(), mod_pipe.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=mod_pipe,
            artifact_path="model_pipeline",
            signature=signature,
            input_example=X_train.to_pandas(),
            registered_model_name="model_pipeline"
        )

        joblib.dump(mod_pipe, "models/mod_pipe.joblib")


if __name__ == "__main__":
    data = pl.read_csv("data/processed/ad_preprocessed.csv", try_parse_dates=True)
    train_model(data)