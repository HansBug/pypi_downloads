"""
PyPI download statistics data loading utilities.

This module provides access to a cached, read-only dataset containing PyPI
download statistics. It ensures that the data file is present locally and can
automatically download a preprocessed dataset from HuggingFace Hub when needed.

The module contains the following main components:

* :func:`load_data` - Load a cached, read-only DataFrame of download statistics.

.. note::
   The cached DataFrame is frozen to prevent accidental mutation. Call
   :meth:`pandas.DataFrame.copy` if you need a mutable copy.

Example::

    >>> from pypi_downloads.data import load_data
    >>> df = load_data()
    >>> df.head()
           name  last_day  last_week  last_month
    0  example     12345     67890      135790

"""

import functools
import os

import numpy as np
import pandas as pd

_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads.parquet')
_HF_REPO = 'HansBug/pypi_downloads'
_HF_FILENAME = 'dataset.parquet'


def _ensure_data_file() -> None:
    """
    Ensure the local data file exists, downloading it if necessary.

    This function checks for the presence of the ``downloads.parquet`` file
    in the package directory. If it is missing, it attempts to download a
    dataset from HuggingFace Hub and writes a filtered version to the local
    file path. The downloaded dataset is filtered to include only valid
    records and the columns ``name``, ``last_day``, ``last_week``, and
    ``last_month``.

    :raises FileNotFoundError: If the data file is missing and cannot be
        downloaded, or if ``huggingface_hub`` is not installed.
    :raises FileNotFoundError: If the auto-download fails for any reason.

    .. note::
       The auto-download feature requires the optional dependency
       ``huggingface_hub`` to be installed.

    Example::

        >>> from pypi_downloads.data import _ensure_data_file
        >>> _ensure_data_file()  # doctest: +SKIP
    """
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
        df = df[df['status'] == 'valid'][['name', 'last_day', 'last_week', 'last_month']].reset_index(drop=True).astype({'last_day': 'int64', 'last_week': 'int64', 'last_month': 'int64'})
        df.to_parquet(_DATA_FILE, index=False)
    except Exception as e:
        raise FileNotFoundError(
            f"Data file not found: {_DATA_FILE!r}\n"
            f"Auto-download from HuggingFace ({_HF_REPO!r}) failed: {e}\n"
            f"To fix this, run `make download_data` from the repository root."
        ) from e


def _freeze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Freeze a DataFrame by making its underlying arrays read-only.

    This function attempts to set the ``writeable`` flag to ``False`` for all
    underlying NumPy arrays in the DataFrame. This helps prevent accidental
    mutation of cached data. For extension arrays (e.g., Pandas nullable
    dtypes), internal NumPy buffers are also frozen when present.

    :param df: DataFrame to freeze.
    :type df: pandas.DataFrame
    :return: The same DataFrame instance with read-only buffers.
    :rtype: pandas.DataFrame

    .. warning::
       This function does not guarantee that all possible data structures are
       immutable, but it significantly reduces the chance of in-place changes.

    Example::

        >>> import pandas as pd
        >>> from pypi_downloads.data import _freeze_dataframe
        >>> df = pd.DataFrame({'a': [1, 2]})
        >>> frozen = _freeze_dataframe(df)
        >>> frozen is df
        True
    """
    for col in df.columns:
        values = df[col].values
        if isinstance(values, np.ndarray):
            values.flags.writeable = False
        else:
            # ExtensionArray (e.g., Int64, Float64): freeze internal NumPy arrays.
            for attr in ('_data', '_mask', '_ndarray'):
                inner = getattr(values, attr, None)
                if isinstance(inner, np.ndarray):
                    inner.flags.writeable = False
    return df


@functools.lru_cache(maxsize=1)
def _load_cached() -> pd.DataFrame:
    """
    Load and cache the download statistics DataFrame.

    This function ensures the data file is present, reads it into a DataFrame,
    freezes the underlying arrays to prevent mutation, and returns the cached
    DataFrame. Subsequent calls return the same cached instance.

    :return: Cached, read-only DataFrame with download statistics.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If the data file is missing and cannot be
        downloaded automatically.

    Example::

        >>> from pypi_downloads.data import _load_cached
        >>> df = _load_cached()
        >>> df is _load_cached()
        True
    """
    _ensure_data_file()
    df = pd.read_parquet(_DATA_FILE)
    return _freeze_dataframe(df)


def load_data(writable: bool = False) -> pd.DataFrame:
    """
    Load PyPI download statistics as a cached DataFrame.

    Reads ``downloads.parquet`` from the package directory on the first call;
    subsequent calls return the same in-memory object without re-reading the
    file.

    If the data file is absent (e.g., when running from a source checkout
    without having run ``make download_data``), an automatic download from
    HuggingFace Hub is attempted when ``huggingface_hub`` is installed.

    :param writable: If ``False`` (default), return the shared cached
        DataFrame whose underlying arrays are frozen (read-only).  If
        ``True``, return a fully writable :meth:`~pandas.DataFrame.copy`
        of the cached data; the copy is independent and may be modified
        freely without affecting the cache.
    :type writable: bool
    :return: DataFrame with columns ``name``, ``last_day``, ``last_week``,
        ``last_month``, containing download statistics for all valid PyPI
        packages.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If ``downloads.parquet`` is missing and cannot
        be downloaded automatically.

    Example::

        >>> from pypi_downloads.data import load_data
        >>> df = load_data()
        >>> df.columns.tolist()
        ['name', 'last_day', 'last_week', 'last_month']
        >>> df_copy = load_data(writable=True)
        >>> df_copy is df
        False
    """
    df = _load_cached()
    return df.copy() if writable else df
