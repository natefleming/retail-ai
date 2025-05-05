import os
import re
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import pyspark.sql.types as T
from datasets import DatasetDict, load_dataset
from pyspark.sql import Row
from pyspark.sql.datasource import DataSource, DataSourceReader, InputPartition


def _dataset_dict_to_pandas(
    self: DatasetDict, split_col: str = "split", splits: Sequence[str] = []
) -> pd.DataFrame:
    """
    Convert a Hugging Face DatasetDict to a pandas DataFrame.

    This helper method extends the DatasetDict class with a to_pandas method through
    monkey patching. It converts multiple dataset splits into a single DataFrame,
    adding a column to identify the source split.

    Args:
        self: The DatasetDict instance
        split_col: Name of the column to store the split information (default: "split")
        splits: Optional list of specific splits to include (empty means include all)

    Returns:
        A pandas DataFrame containing all splits (or specified splits) with a
        column identifying the source split
    """
    datasets: list[pd.DataFrame] = []
    for split_name, split_dataset in self.items():
        if splits and split_name not in splits:
            continue
        split_dataset = split_dataset.to_pandas()
        split_dataset[split_col] = split_name
        datasets.append(split_dataset)

    if datasets:
        return pd.concat(datasets, ignore_index=True)
    else:
        return pd.DataFrame()


# Monkey patch the DatasetDict class to add the to_pandas method
DatasetDict.to_pandas = _dataset_dict_to_pandas


# List of supported configuration options for the Hugging Face data source
allowed_options: Sequence[str] = (
    "repo_id",  # Hugging Face dataset repository ID
    "url",  # Alternative URL for loading datasets
    "cache_dir",  # Local directory to cache datasets
    "split_col",  # Column name for storing split information
    "splits",  # Comma-separated list of splits to include
    "primary_key",  # Field to use as primary key
)


def _validate_options(options: dict[str, str]) -> None:
    """
    Validate the options provided to the Hugging Face data source.

    Ensures that all required options are provided and that they meet
    the necessary criteria for successful dataset loading.

    Args:
        options: Dictionary of configuration options

    Raises:
        ValueError: If any validation checks fail
    """
    # Check if all provided options are valid
    for option in options.keys():
        if option not in allowed_options:
            raise ValueError(f"Invalid option: {option}")

    # Check for required options and their combinations
    has_repo_id: bool = "repo_id" in options
    has_url: bool = "url" in options
    has_cache_dir: bool = (
        "cache_dir" in options and "HF_DATASETS_CACHE" not in os.environ
    )

    # Either repo_id or url is required
    if not has_repo_id and not has_url:
        raise ValueError("Either repo_id or url is required")

    # Cannot have both repo_id and url
    if has_url and has_repo_id:
        raise ValueError("Either repo_id or url is required, not both")

    # URL must have proper prefix
    if has_url and not re.match(r"^(hf://|https?://)", options["url"]):
        raise ValueError("url must start with hf://, http://, or https://")

    # Cache directory is required for repo_id
    if has_repo_id and not has_cache_dir:
        raise ValueError(
            "cache_dir is required or HF_DATASETS_CACHE environment variable is required"
        )


def _repo_id_from(options: dict[str, str]) -> str:
    """Extract and return the repository ID from options."""
    repo_id: str = options.get("repo_id")
    return repo_id


def _url_from(options: dict[str, str]) -> str:
    """Extract and return the URL from options."""
    url: str = options.get("url")
    return url


def _cache_dir_from(options: dict[str, str]) -> str:
    """
    Extract and return the cache directory from options or environment.

    Uses the cache_dir option if available, otherwise falls back to the
    HF_DATASETS_CACHE environment variable.
    """
    cache_dir: str = options.get("cache_dir", os.environ.get("HF_DATASETS_CACHE"))
    return cache_dir


def _primary_key_from(options: dict[str, str]) -> str | None:
    """Extract and return the primary key from options, converting to lowercase."""
    return options.get("primary_key", "").lower()


def _split_col_from(options: dict[str, str]) -> str:
    """Extract and return the split column name from options."""
    return options.get("split_col", "split")


def _splits_from(options: dict[str, str]) -> Sequence[str]:
    """
    Extract and return the list of splits from options.

    Converts a comma-separated string of split names into a list,
    removing any empty entries.
    """
    splits: str = options.get("splits", "")
    return [s.strip() for s in splits.split(",") if s.strip()]


def _load_dataset(
    url: str,
    repo_id: str,
    cache_dir: str,
    split_col: str = "split",
    splits: Sequence[str] = [],
) -> pd.DataFrame:
    """
    Load a dataset from Hugging Face or a URL into a pandas DataFrame.

    Handles loading datasets from Hugging Face repositories or from URLs
    pointing to supported file formats (parquet, csv).

    Args:
        url: URL to a dataset file (used if repo_id is not provided)
        repo_id: Hugging Face dataset repository ID
        cache_dir: Local directory to cache datasets
        split_col: Column name for storing split information
        splits: List of specific splits to include

    Returns:
        A pandas DataFrame containing the loaded dataset

    Raises:
        ValueError: If the URL points to an unsupported file type
    """
    if cache_dir:
        _ = Path(cache_dir).mkdir(parents=True, exist_ok=True)

    pdf: pd.DataFrame

    if repo_id:
        # Load from Hugging Face repository
        datasets: DatasetDict = load_dataset(path=repo_id, cache_dir=cache_dir)
        pdf = datasets.to_pandas(split_col=split_col, splits=splits)
    else:
        # Load from URL based on file extension
        if url.endswith(".parquet"):
            pdf = pd.read_parquet(url)
        elif url.endswith(".csv"):
            print("Loading csv...")
            pdf = pd.read_csv(url, sep="\t")
        else:
            raise ValueError(f"Unsupported file type: {url}")
    return pdf


class HuggingfaceDataSource(DataSource):
    """
    A custom PySpark DataSource implementation for loading datasets from Hugging Face.

    This DataSource allows Spark to read data directly from Hugging Face datasets
    or from URLs pointing to supported file formats (parquet, csv). It integrates
    with Spark's data source API to provide schema inference and data reading.
    """

    @classmethod
    def name(cls):
        """Return the name of this data source for registration with Spark."""
        return "huggingface"

    def schema(self) -> T.StructType | str:
        """
        Infer and return the schema for the dataset.

        This method loads a sample of the dataset to infer the schema,
        converting pandas dtypes to Spark SQL types and handling special
        cases like primary keys.

        Returns:
            A PySpark StructType representing the dataset schema
        """
        repo_id: str = _repo_id_from(self.options)
        url: str = _url_from(self.options)
        cache_dir: str = _cache_dir_from(self.options)
        split_col: str = _split_col_from(self.options)
        splits: Sequence[str] = _splits_from(self.options)
        primary_key: str = _primary_key_from(self.options)

        # Load sample data to infer schema
        pdf: pd.DataFrame = _load_dataset(
            url=url,
            repo_id=repo_id,
            cache_dir=cache_dir,
            split_col=split_col,
            splits=splits,
        )

        def infer_type(dtype) -> T.DataType:
            """Convert pandas dtype to corresponding Spark SQL type."""
            if pd.api.types.is_integer_dtype(dtype):
                return T.LongType()
            elif pd.api.types.is_float_dtype(dtype):
                return T.DoubleType()
            elif pd.api.types.is_bool_dtype(dtype):
                return T.BooleanType()
            else:
                return T.StringType()

        def sanitize_name(name):
            """
            Sanitize column names to be compatible with Spark SQL.

            Replaces non-alphanumeric characters with underscores and
            ensures column names start with a letter or underscore.
            """
            name = re.sub(r"[^\w]", "_", str(name))
            if not re.match(r"^[a-zA-Z_]", name):
                name = "col_" + name
            return name

        # Create struct fields for each column, marking primary key as non-nullable
        struct_fields: Sequence[T.StructField] = [
            T.StructField(
                sanitize_name(col), infer_type(dtype), primary_key != col.lower()
            )
            for col, dtype in pdf.dtypes.items()
        ]

        return T.StructType(struct_fields)

    def reader(self, schema: T.StructType) -> DataSourceReader:
        """
        Create and return a reader for this data source.

        Args:
            schema: The inferred or user-provided schema

        Returns:
            A HuggingfaceDataSourceReader to read the data
        """
        return HuggingfaceDataSourceReader(schema, self.options)


class HuggingfaceDataSourceReader(DataSourceReader):
    """
    A reader for the HuggingfaceDataSource that handles actual data loading.

    This class implements the DataSourceReader interface required by Spark's
    data source API, providing methods to read data from Hugging Face datasets
    or URLs into Spark's internal representation.
    """

    def __init__(self, schema: T.StructType, options: dict[str, str]):
        """
        Initialize the reader with schema and options.

        Args:
            schema: The schema to use for reading data
            options: Configuration options for the data source
        """
        _validate_options(options)
        self.schema: T.StructType = schema
        self.repo_id: str = _repo_id_from(options)
        self.url: str = _url_from(options)
        self.cache_dir: str = _cache_dir_from(options)
        self.split_col: str = _split_col_from(options)
        self.splits: Sequence[str] = _splits_from(options)

    def read(self, partition: InputPartition) -> Iterator[tuple] | Iterator[Row]:
        """
        Read data from the dataset and yield rows.

        This method loads the dataset into a pandas DataFrame and then
        yields rows one at a time to be converted into Spark Row objects.

        Args:
            partition: Input partition (not used as the entire dataset is loaded)

        Returns:
            An iterator of tuples or Row objects representing dataset rows
        """
        # Load the entire dataset
        pdf: pd.DataFrame = _load_dataset(
            url=self.url,
            repo_id=self.repo_id,
            cache_dir=self.cache_dir,
            split_col=self.split_col,
            splits=self.splits,
        )

        # Yield rows one at a time
        for row in pdf.itertuples(index=False, name=None):
            yield row
