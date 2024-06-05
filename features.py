import polars as pl

from logger import logger


def remove_country_all(dataf: pl.DataFrame) -> pl.DataFrame:
    """
    Remove rows where 'country' is 'ALL' from the DataFrame.

    Parameters
    ----------
    dataf : polars.DataFrame
        The DataFrame from which to remove the rows.

    Returns
    -------
    polars.DataFrame
        The DataFrame with the rows where 'country' is 'ALL' removed.
    """
    logger.info("Removing country ALL")
    return dataf.filter(pl.col("country") != "ALL")


def sum_duplicates(dataf: pl.DataFrame) -> pl.DataFrame:
    """
    Sum the duplicate rows in the DataFrame.

    Parameters
    ----------
    dataf : polars.DataFrame
        The DataFrame from which to sum the duplicate rows.

    Returns
    -------
    polars.DataFrame
        The DataFrame with the duplicate rows summed.
    """
    logger.info("Summing duplicate rows")
    return (dataf.group_by(["os", "country", "date"])
            .agg([pl.sum("count").alias("count")])
            .select(["date", "country", "os", "count"]))


def add_timely_features(dataf: pl.DataFrame) -> pl.DataFrame:
    """
    Add timely features to the DataFrame.

    Parameters
    ----------
    dataf : polars.DataFrame
        The DataFrame to add the timely features to.

    Returns
    -------
    polars.DataFrame
        The DataFrame with the timely features added.
    """
    logger.info("Adding timely features")
    return dataf.with_columns([
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("weekday"),
        pl.col("date").dt.ordinal_day().alias("ordinal_day")])


if __name__ == "__main__":
    logger.info("Loading raw data")
    df = pl.scan_csv("data/raw/anomaly_detection_dataset.csv",
                     try_parse_dates=True)
    df_processed = (
        df
        .pipe(remove_country_all)
        .pipe(sum_duplicates)
        # .pipe(add_timely_features)
        .sort(by=["date", "country", "os"])
        .collect()
    )
    logger.info("Saving processed data")
    df_processed.write_csv("data/processed/ad_preprocessed.csv")
