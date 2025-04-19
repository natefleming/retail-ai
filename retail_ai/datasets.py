from typing import Iterator, Sequence
import os
import re
import time
from pathlib import Path

from pyspark.sql.datasource import DataSource, DataSourceReader, SimpleDataSourceStreamReader, InputPartition
from pyspark.sql import SparkSession, DataFrame, Row
import pyspark.sql.types as T
import pyspark.sql.functions as F

import pandas as pd

from datasets import load_dataset, DatasetDict



def _dataset_dict_to_pandas(
    self: DatasetDict, 
    split_col: str = "split", 
    splits: Sequence[str] = []
) -> pd.DataFrame:
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


# monkey patch
DatasetDict.to_pandas = _dataset_dict_to_pandas  


allowed_options: Sequence[str] = (
    "repo_id", 
    "cache_dir", 
    "split_col",
    "splits",
    "primary_key"
)


def _validate_options(options: dict[str, str]) -> None:
    for option in options.keys():
        if option not in allowed_options:
            raise ValueError(f"Invalid option: {option}")


def _repo_id_from(options: dict[str, str]) -> str:
    repo_id: str = options.get("repo_id")
    if repo_id is None:
        raise ValueError("repo_id is required")
    return repo_id
    

def _cache_dir_from(options: dict[str, str]) -> str:
    cache_dir: str = options.get("cache_dir", os.environ.get("HF_DATASETS_CACHE"))
    if cache_dir is None:
        raise ValueError("cache_dir is required or HF_DATASETS_CACHE environment variable is required")
    return cache_dir


def _primary_key_from(options: dict[str, str]) -> str | None:
    return options.get("primary_key", "").lower()

def _split_col_from(options: dict[str, str]) -> str:
    return options.get("split_col", "split")


def _splits_from(options: dict[str, str]) -> Sequence[str]:
    splits: str = options.get("splits", "")
    return [s.strip() for s in splits.split(",") if s.strip()]


def _load_dataset(
    repo_id: str, 
    cache_dir: str, 
    split_col: str = "split",
    splits: Sequence[str] = [],
) -> pd.DataFrame:
    _ = Path(cache_dir).mkdir(parents=True, exist_ok=True)

    datasets: DatasetDict = load_dataset(
        path=repo_id, 
        cache_dir=cache_dir
    )
    
    pdf: pd.DataFrame = datasets.to_pandas(split_col=split_col, splits=splits)

    return pdf



class HuggingfaceDataSource(DataSource):

    @classmethod
    def name(cls):
        return "huggingface"

    def schema(self) -> T.StructType | str:
        repo_id: str = _repo_id_from(self.options)
        cache_dir: str = _cache_dir_from(self.options)
        split_col: str = _split_col_from(self.options)
        splits: Sequence[str] = _splits_from(self.options)
        primary_key: str = _primary_key_from(self.options)

        pdf: pd.DataFrame = _load_dataset(
            repo_id=repo_id, 
            cache_dir=cache_dir, 
            split_col=split_col,
            splits=splits,
        )

        def infer_type(dtype) -> T.DataType:
            if pd.api.types.is_integer_dtype(dtype):
                return T.LongType()
            elif pd.api.types.is_float_dtype(dtype):
                return T.DoubleType()
            elif pd.api.types.is_bool_dtype(dtype):
                return T.BooleanType()
            else:
                return T.StringType()

        def sanitize_name(name):
            name = re.sub(r'[^\w]', '_', str(name))
            if not re.match(r'^[a-zA-Z_]', name):
                name = 'col_' + name
            return name
    
        struct_fields: Sequence[T.StructField] = [
            T.StructField(sanitize_name(col), infer_type(dtype), primary_key != col.lower())
            for col, dtype in pdf.dtypes.items()
        ]
        
        return T.StructType(struct_fields)

    def reader(self, schema: T.StructType) -> DataSourceReader:
        return HuggingfaceDataSourceReader(schema, self.options)
    


class HuggingfaceDataSourceReader(DataSourceReader):

    def __init__(self, schema: T.StructType, options: dict[str, str]):
        _validate_options(options)
        self.schema: T.StructType = schema
        self.repo_id: str = _repo_id_from(options)
        self.cache_dir: str = _cache_dir_from(options)
        self.split_col: str = _split_col_from(options)
        self.splits: Sequence[str] = _splits_from(options)

    def read(self, partition: InputPartition) -> Iterator[tuple] | Iterator[Row]:

        pdf: pd.DataFrame = _load_dataset(
            repo_id=self.repo_id, 
            cache_dir=self.cache_dir, 
            split_col=self.split_col,
            splits=self.splits,
        )

        for row in pdf.itertuples(index=False, name=None):
            yield row

