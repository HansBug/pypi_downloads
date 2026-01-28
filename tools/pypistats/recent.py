"""
Module for retrieving recent download statistics from PyPI Stats API.

This module provides functionality to fetch recent download statistics for Python packages
from the PyPI Stats service (pypistats.org). It can retrieve download counts for the last
day, week, and month for any given package.
"""

import json
from typing import Optional

import requests

from ..utils import get_requests_session


def get_pypistats_recent(name: str, session: Optional[requests.Session] = None) -> dict:
    """
    Get recent download statistics for a PyPI package.

    This function retrieves recent download statistics from pypistats.org for a specified
    Python package. The statistics include download counts for the last day, last week,
    and last month.

    :param name: The name of the PyPI package to query.
    :type name: str
    :param session: Optional requests session to use for the HTTP request. If None, a new session will be created.
    :type session: Optional[requests.Session]

    :return: A dictionary containing the package download statistics with the following structure:
        - 'data': dict with keys 'last_day', 'last_week', 'last_month' containing download counts
        - 'package': str, the package name
        - 'type': str, the type of statistics (typically 'recent_downloads')
    :rtype: dict

    :raises requests.HTTPError: If the HTTP request fails or returns an error status code.

    Example::
        >>> get_pypistats_recent('numpy')  # Get statistics for numpy package
        {
            "data": {
                "last_day": 25704278,
                "last_month": 632261505,
                "last_week": 156080250
            },
            "package": "numpy",
            "type": "recent_downloads"
        }

        >>> # Using a custom session
        >>> session = requests.Session()
        >>> get_pypistats_recent('lightzero', session=session)
        {
            "data": {
                "last_day": 5,
                "last_month": 250,
                "last_week": 45
            },
            "package": "lightzero",
            "type": "recent_downloads"
        }
    """
    session = session or get_requests_session()
    resp = session.get(f'https://pypistats.org/api/packages/{name}/recent')
    resp.raise_for_status()
    return resp.json()


if __name__ == '__main__':
    pypi_name = 'numpy'
    print(f'Information of {pypi_name!r}:')
    print(json.dumps(get_pypistats_recent(pypi_name), indent=4, sort_keys=True))
