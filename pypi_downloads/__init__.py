"""
PyPI Downloads Package.

This package provides access to cached PyPI download statistics and exposes a
simple, stable public API for retrieving the dataset.

The package contains the following main components:

* :func:`load_data` - Load the cached PyPI download statistics dataset.
* :data:`__version__` - Package version string.

Example::

    >>> import pypi_downloads
    >>> pypi_downloads.__version__
    '0.0.1'
    >>> df = pypi_downloads.load_data()
    >>> df.columns.tolist()
    ['name', 'last_day', 'last_week', 'last_month']

"""

from .config.meta import __VERSION__ as __version__
from .data import load_data
