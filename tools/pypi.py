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
    for aitem in page('a').items():
        name = aitem.text().strip()
        url = urljoin(resp.url, aitem.attr('href'))
        items.append(PypiItem(name, url))

    return items


if __name__ == '__main__':
    print("Fetching PyPI package index...")

    # Get all packages from PyPI
    packages = get_pypi_index()

    # Display total count
    print(f"\nTotal packages found: {len(packages):,}")

    # Show some well-known packages
    print("\nLooking for some popular packages:")
    popular_packages = ['numpy', 'pandas', 'requests', 'django', 'flask']
    found_popular = []

    for package in packages:
        if package.name.lower() in popular_packages:
            found_popular.append(package)
            print(f"✓ Found: {package.name} -> {package.url}")

    # Display first 10 packages as examples
    print(f"\nFirst 10 packages (alphabetically):")
    for i, package in enumerate(packages[:10]):
        print(f"{i + 1:2d}. {package.name}")

    # Create pandas DataFrame for better visualization
    try:
        import pandas as pd

        # Convert to DataFrame
        df = pd.DataFrame([
            {'Package Name': item.name, 'URL': item.url}
            for item in packages
        ])

        print(f"\nPandas DataFrame Info:")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())

        print(f"\nLast 5 rows:")
        print(df.tail())

        # Some statistics
        print(f"\nPackage name length statistics:")
        df['name_length'] = df['Package Name'].str.len()
        print(df['name_length'].describe())

        # Show packages with shortest and longest names
        shortest_name = df.loc[df['name_length'].idxmin()]
        longest_name = df.loc[df['name_length'].idxmax()]

        print(f"\nShortest package name: '{shortest_name['Package Name']}' ({shortest_name['name_length']} chars)")
        print(f"Longest package name: '{longest_name['Package Name']}' ({longest_name['name_length']} chars)")

    except ImportError:
        print("\nPandas not installed. Install with: pip install pandas")
        print("Showing raw data structure instead:")
        print(f"Sample package objects:")
        for i, package in enumerate(packages[:3]):
            print(f"{i + 1}. PypiItem(name='{package.name}', url='{package.url}')")
