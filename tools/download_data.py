"""
Download and prepare the PyPI download statistics dataset.

Fetches ``dataset.parquet`` from a HuggingFace Hub dataset repository,
filters to packages with ``status == 'valid'``, casts download-count columns
to ``int64``, and writes the result to the package data directory.

Usage::

    python -m tools.download_data
    python -m tools.download_data --repo other/repo --output /path/to/out.parquet

"""

import argparse
import os

import pandas as pd
from huggingface_hub import hf_hub_download

_DEFAULT_REPO = 'HansBug/pypi_downloads'
_DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'pypi_downloads', 'downloads.parquet',
)
_INT_COLS = ['last_day', 'last_week', 'last_month']


def download_data(repo: str = _DEFAULT_REPO, output: str = _DEFAULT_OUTPUT) -> None:
    """Download, filter, and save the PyPI download statistics parquet file.

    :param repo: HuggingFace Hub dataset repository ID to download from.
    :type repo: str
    :param output: Destination path for the filtered ``downloads.parquet``.
    :type output: str
    """
    src = hf_hub_download(repo_id=repo, repo_type='dataset', filename='dataset.parquet')
    df = pd.read_parquet(src)
    df = (
        df[df['status'] == 'valid'][['name'] + _INT_COLS]
        .reset_index(drop=True)
        .astype({col: 'int64' for col in _INT_COLS})
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_parquet(output, index=False)
    print(f'Saved {len(df):,} records to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', '-r', default=_DEFAULT_REPO,
                        help='HuggingFace dataset repo ID (default: %(default)s)')
    parser.add_argument('--output', '-o', default=_DEFAULT_OUTPUT,
                        help='Output parquet path (default: pypi_downloads/downloads.parquet)')
    args = parser.parse_args()
    download_data(repo=args.repo, output=args.output)
