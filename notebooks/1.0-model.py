# %%
import polars as pl
import altair as alt

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklego.mixture import GMMOutlierDetector
# %%
df = pl.scan_csv("../data/processed/anomaly_detection_dataset_preprocessed.csv",
                 try_parse_dates=True)

# %%
df.tail().collect()

# %%
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

# %%
feat_pipe = make_column_transformer(
    (OneHotEncoder(sparse_output=False), ["country"]),
    (OrdinalEncoder(categories=[["Y", "X", "W", "T"]]), ["os"]),
    (StandardScaler(), ["count"])
)

mod_pipe = make_pipeline(
    feat_pipe,
    GMMOutlierDetector(n_components=4*7, threshold=0.99)
)

mod_pipe.fit(X_train)

# %%
alt.Chart(X_train.with_columns(
    pl.Series(values=mod_pipe.predict(X_train) == -1, name="anomaly")
    )).mark_tick(size=40, thickness=2, color="black", opacity=0.75).encode(
    x='count:Q',
    y='os:N',
    column='country:N',
    color=alt.condition(alt.expr.datum["anomaly"],
                                alt.ColorValue('black'),
                                "os:N")
).properties(
    height=300,
    title="Count values per country and os for the train set"
).interactive()

# %%
alt.Chart(X_test.with_columns(
    pl.Series(values=mod_pipe.predict(X_test) == -1, name="anomaly")
    )).mark_tick(size=40, thickness=2, color="black", opacity=0.75).encode(
    x='count:Q',
    y='os:N',
    column='country:N',
    color=alt.condition(alt.expr.datum["anomaly"],
                                alt.ColorValue('black'),
                                "os:N")
).properties(
    height=300,
    title="Count values (and Anomalies) per country and os for the test set"
).interactive()
# %%
