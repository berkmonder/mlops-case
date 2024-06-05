# %%
import hvplot.polars
import polars as pl
# import hiplot as hip

# %%
hvplot.help('hist')
pl.Config.set_tbl_rows(25)

# %%
df_raw = pl.scan_csv("../data/raw/anomaly_detection_dataset.csv", try_parse_dates=True)

# %%
df_raw.head().collect()

# %%
df_raw.collect().describe()

# %%
# To check if there are rows with negative count
df_raw.filter(pl.col("count") < 0).collect()

# %%
# To check if there are duplicate rows
df_raw.filter(pl.struct(["date", "country", "os"]).is_duplicated()).collect()

# %%
df_raw.group_by(["os"]).len().collect()
# %%
df_raw.group_by(["country"]).len().collect()

# %%
df_raw.filter(pl.col("country") == "ALL").collect()

# %%
df_raw.null_count().collect()

# %%
df = (
    df_raw
    .filter(pl.col("country") != "ALL")
    .group_by(["os", "country", "date"])
            .agg([pl.sum("count").alias("count")])
            .select(["date", "country", "os", "count"])
    .with_columns([
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("weekday"),
        pl.col("date").dt.ordinal_day().alias("ordinal_day")])
    .sort(by=["date", "country", "os"])
)

# %%
df.head().collect()

# %%
df.select(pl.all().approx_n_unique()).collect()

# %%
df.drop("date").collect().to_pandas().to_csv("../data/interim/hiplot.csv", index=False)
# hip.Experiment.from_dataframe(df.collect().to_pandas().to_csv("./data/hiplot.csv")).display()

# %%
(
    df
    .collect()
    .plot
    .scatter(x="date", y="count", c="os")
)

# %%
(
    df
    .collect()
    .plot
    .scatter(x="date", y="count", color="country")
)

# %%
(
    df
    .collect()
    .plot
    .scatter(x="country", y="count", color="os")
)

# %%
(
    df
    .collect()
    .plot
    .scatter(x="os", y="count", color="country")
)

# %%
country = "XX" # ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX"]
(
    df
    .filter(pl.col("country") == country)
    .collect()
    .plot
    .hist(
        y="count", by="os", bins=100, xformatter='%.0f')
    .opts(
        title=f"Count distribution for {country} by OS",
    )
)

# %%
os = "T" # ["X", "Y", "W", "T"]
(
    df
    .filter(pl.col("os") == os)
    .collect()
    .plot
    .hist(
        y="count", by="country", bins=100, xformatter='%.0f')
    .opts(
        title=f"Count distribution for {os} by Country",
    )
)

# %%
aggregate_function = "sum" # ["sum", "count", "min", "mean", "median", "max"]
(
    df
    .collect()
    .pivot(values="count", index="date", columns="os", aggregate_function=aggregate_function)
    .plot
    .line(x="date", value_label="count", yformatter='%.0f')
)

# %%
aggregate_function = "sum" # ["sum", "count", "min", "mean", "median", "max"]
(
    df
    .collect()
    .pivot(values="count", index="date", columns="country", aggregate_function=aggregate_function)
    .plot
    .line(x="date", value_label="count", yformatter='%.0f')
)

# %%
(
    df
    .filter(pl.col("os") == "X") # ["X", "Y", "W", "T"]
    # .filter(pl.col("country") == "XX") # ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX"]
    .collect()
    .pivot(values="count", index="date", columns=["country", "os"])
    .plot
    .line(
        x="date",
        logy=True,
        value_label="count", yformatter='%.0f')
)
# %%
