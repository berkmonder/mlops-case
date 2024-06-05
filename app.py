import os

import joblib
import polars as pl
from fastapi import FastAPI, Response

from logger import logger
from model_train import train_model

app = FastAPI()

@app.get("/")
async def read_root() -> Response:
    return Response("The server is running.")


@app.get("/predict")
async def predict(Country: str, OS: str, Count: int) -> int:
    """
    Predict the anomaly for the given country, os and count values using the trained model pipeline.

    Parameters
    ----------
    Country : str
        The country
    OS : str
        The operating system
    Count : int
        The count value
    
    Returns
    -------
    int
        The anomaly prediction (-1 or 1)
    """
    logger.info("Received request to predict")
    if not os.path.exists("models/mod_pipe.joblib"):
        logger.info("Model pipeline not found. Training model...")
        df = pl.read_csv("data/processed/ad_preprocessed.csv", try_parse_dates=True)
        train_model(df)
    mod_pipe = joblib.load("models/mod_pipe.joblib")
    logger.info("Model pipeline loaded")
    X = pl.DataFrame(data={"date": "null", "country": [Country], "os": [OS], "count": [Count]})
    y_pred = mod_pipe.predict(X)
    logger.info(f"Anomaly prediction: {y_pred[0]}")
    return y_pred[0]


@app.get("/retrain")
async def retrain(n_components: int = 4*7, threshold: float = 0.99) -> dict:
    logger.info("Received request to retrain")
    df = pl.read_csv("data/processed/ad_preprocessed.csv", try_parse_dates=True)
    train_model(df, n_components=n_components, threshold=threshold)
    logger.info(f"Model retrained with n_components={n_components} and threshold={threshold}")
    return {"message": "Model retrained"}
