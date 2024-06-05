# %%
import numpy as np
import polars as pl
import altair as alt

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklego.mixture import GMMOutlierDetector
from sklearn.mixture import GaussianMixture
# %%
df = pl.scan_csv("../data/processed/ad_preprocessed.csv",
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

# %%
alt.Chart(X_test.with_columns(
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

# %%
X_test.filter(
    pl.col("country") == "ZX",
    pl.col("os") == "Y",
    pl.col("count") <= 150_000
    ).to_pandas()

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    return accuracy, precision, recall, f1

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


# %%
y_pred = mod_pipe.predict(X_test_sample)
eval_metrics(y_sample_true, y_pred)
# %%
