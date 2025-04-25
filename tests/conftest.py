import logging
import os
import sys
import tempfile
from pathlib import Path


import pytest


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYARROW_IGNORE_TIMEZONE"] = str(1)
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

src_dir: Path = Path(__file__).parents[1] 
test_dir: Path = Path(__file__).parents[0]
sys.path.insert(0, str(test_dir.resolve()))
sys.path.insert(0, str(src_dir.resolve()))
