import functools
import os

import numpy as np
import pandas as pd

_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads.parquet')
_HF_REPO = 'HansBug/pypi_downloads'
_HF_FILENAME = 'dataset.parquet'


def _ensure_data_file():
    if os.path.exists(_DATA_FILE):
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"Data file not found: {_DATA_FILE!r}\n"
            f"The package is designed for offline use and ships with a pre-built data file, "
            f"but it is missing. huggingface_hub is not installed, so auto-download is unavailable.\n"
            f"To fix this, either:\n"
            f"  - Run `make download_data` from the repository root, or\n"
            f"  - Install huggingface_hub to enable auto-download: pip install huggingface_hub"
        )

    try:
        src = hf_hub_download(repo_id=_HF_REPO, repo_type='dataset', filename=_HF_FILENAME)
        df = pd.read_parquet(src)
        df = df[df['status'] == 'valid'][['name', 'last_day', 'last_week', 'last_month']].reset_index(drop=True)
        df.to_parquet(_DATA_FILE, index=False)
    except Exception as e:
        raise FileNotFoundError(
            f"Data file not found: {_DATA_FILE!r}\n"
            f"Auto-download from HuggingFace ({_HF_REPO!r}) failed: {e}\n"
            f"To fix this, run `make download_data` from the repository root."
        ) from e


def _freeze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        values = df[col].values
        if isinstance(values, np.ndarray):
            values.flags.writeable = False
        else:
            # ExtensionArray (e.g. Int64, Float64): freeze internal numpy arrays
            for attr in ('_data', '_mask', '_ndarray'):
                inner = getattr(values, attr, None)
                if isinstance(inner, np.ndarray):
                    inner.flags.writeable = False
    return df


@functools.lru_cache(maxsize=1)
def _load_cached() -> pd.DataFrame:
    _ensure_data_file()
    df = pd.read_parquet(_DATA_FILE)
    return _freeze_dataframe(df)


def load_data() -> pd.DataFrame:
    """Load PyPI download statistics as a read-only cached DataFrame.

    Reads ``downloads.parquet`` from the package directory on the first call;
    subsequent calls return the same in-memory object without re-reading the
    file.  The returned DataFrame is frozen (underlying numpy arrays are set to
    read-only) to prevent accidental mutation of the cached object.  If you
    need to modify the data, call ``.copy()`` on the result first.

    If the data file is absent (e.g. when running from a source checkout
    without having run ``make download_data``), an automatic download from
    HuggingFace Hub is attempted when ``huggingface_hub`` is installed.

    :return: DataFrame with columns ``name``, ``last_day``, ``last_week``,
        ``last_month``, containing download statistics for all valid PyPI
        packages.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: if ``downloads.parquet`` is missing and cannot
        be downloaded automatically.
    """
    return _load_cached()
