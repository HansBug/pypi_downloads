from dataclasses import dataclass
from typing import Optional, List
from urllib.parse import urljoin

import requests
from pyquery import PyQuery as pq

DEFAULT_INDEX_URL = 'https://pypi.org/simple'


@dataclass
class PypiItem:
    name: str
    url: str


def get_pypi_index(index_url: Optional[str] = None) -> List[PypiItem]:
    index_url = index_url or DEFAULT_INDEX_URL
    resp = requests.get(index_url)
    resp.raise_for_status()

    page = pq(resp.text)
    items = []
    for aitem in page('a'):
        name = aitem.text().strip()
        url = urljoin(resp.url, aitem.attr('href'))
        items.append(PypiItem(name, url))

    return items


if __name__ == '__main__':
    pass
