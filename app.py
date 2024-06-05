import os
import json

import joblib
import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from logger import logger
from model_train import train_model

app = FastAPI()

@app.get("/")
async def read_root() -> Response:
    return Response("The server is running.")


@app.get("/predict")
async def predict(Country: str, OS: str, Count: int) -> dict:
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
    dict
        The anomaly prediction
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
    return {"y_pred": y_pred[0]}


@app.get("/retrain")
async def retrain() -> dict:
    logger.info("Received request to retrain")
    return {"message": "Model retrained"}
