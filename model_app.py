import os

import joblib
import polars as pl
import altair as alt
import streamlit as st
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklego.mixture import GMMOutlierDetector

from model_train import train_model

st.title("Anomaly Detection of the number of API requests made on that date from that county and OS combination")

st.markdown("---")

st.markdown("### Here is an overview of the dataset:")
df = pl.scan_csv("./data/processed/ad_preprocessed.csv",
                 try_parse_dates=True)
st.dataframe(df.collect().head())

st.markdown("---")

X_train = (
    df
    .filter(
        pl.col("date") < pl.datetime(2024, 4, 1),
        # pl.col("country") == "XX"
        ).collect())

X_test = (
    df
    .filter(
        pl.col("date") >= pl.datetime(2024, 4, 1),
        # pl.col("country") == "XX"
        ).collect())

feat_pipe = make_column_transformer(
    (OneHotEncoder(sparse_output=False), ["country"]),
    (OrdinalEncoder(categories=[["Y", "X", "W", "T"]]), ["os"]),
    (StandardScaler(), ["count"])
)

mod_pipe = make_pipeline(
    feat_pipe,
    GMMOutlierDetector(n_components=4*7, threshold=0.99)
)

st.write(mod_pipe.fit(X_train))

st.markdown("---")

chart_train = alt.Chart(X_train.with_columns(
    pl.Series(values=mod_pipe.predict(X_train) == -1, name="anomaly")
    )).mark_tick(size=40, thickness=2, color="black", opacity=0.75).encode(
    x="count:Q",
    y="os:N",
    column="country:N",
    color=alt.condition(alt.expr.datum["anomaly"],
                                alt.ColorValue("black"),
                                "os:N")
).properties(
    height=300,
    title="Count values per country and os for the train set"
).interactive()

st.altair_chart(chart_train)

st.markdown("---")

chart_test = alt.Chart(X_test.with_columns(
    pl.Series(values=mod_pipe.predict(X_test) == -1, name="anomaly")
    )).mark_tick(size=40, thickness=2, color="black", opacity=0.75).encode(
    x="count:Q",
    y="os:N",
    column="country:N",
    color=alt.condition(alt.expr.datum["anomaly"],
                                alt.ColorValue("black"),
                                "os:N")
).properties(
    height=300,
    title="Count values (and Anomalies) per country and os for the test set"
).interactive()

st.altair_chart(chart_test)