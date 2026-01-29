import logging
import os.path
import time
from threading import Lock
from typing import Optional

import pandas as pd
from hbutils.concurrent import parallel_call
from hbutils.logging import ColoredFormatter
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import number_to_tag
from natsort import natsorted

from .pypi import get_pypi_index
from .pypistats.recent import get_pypistats_recent
from .utils import get_requests_session


def sync(repository: str, proxy_pool: Optional[str] = None, deploy_span: float = 5 * 60):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    session = get_requests_session()
    if proxy_pool:
        logging.info(f'Proxy pool {proxy_pool!r} enabled.')
        session.proxies.update({
            'http': proxy_pool,
            'https': proxy_pool
        })

    logging.info('Getting index ...')
    d_index = {item.name: item for item in get_pypi_index(session=session)}

    if hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='dataset.parquet'
    ):
        logging.info('Load from repository ...')
        df = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='dataset.parquet'
        ))
        df = df[df['name'].isin(d_index)]
        names = set(df['name'])
        records = df.to_dict('records')
    elif hf_client.file_exists(
            repo_id=repository,
            repo_type='dataset',
            filename='dataset.csv'
    ):
        logging.info('Load from repository ...')
        df = pd.read_csv(hf_client.hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='dataset.csv'
        ))
        df = df[df['name'].isin(d_index)]
        names = set(df['name'])
        records = df.to_dict('records')
    else:
        logging.info(f'No existing file found.')
        names = set()
        records = []

    for name, item in d_index.items():
        if name not in names:
            records.append({
                'name': name,
                'url': item.url,
                'last_day': None,
                'last_month': None,
                'last_week': None,
                'is_empty': None,
                'updated_at': None,
            })

    d_records = {item['name']: item for item in records}

    df_x = pd.DataFrame(records)
    if 'is_empty' not in df_x.columns:
        df_x['is_empty'] = None
    df_x = df_x[
        df_x['updated_at'].isnull() |
        (~df_x['updated_at'].isnull() & (df_x['updated_at'] + 30 * 86400 < time.time()))
        ]
    logging.info(f'Records to refresh:\n{df_x}')

    has_update = False
    last_saved_at = None
    lock = Lock()

    def _deploy(force=False):
        nonlocal has_update, last_saved_at
        if not has_update:
            return
        if not force and last_saved_at is not None and last_saved_at + deploy_span > time.time():
            return

        with TemporaryDirectory() as upload_dir:
            dst_parquet_file = os.path.join(upload_dir, 'table.parquet')
            logging.info(f'Saving to {dst_parquet_file}')
            df = pd.DataFrame(list(d_records.values()))
            df = df.sort_values(by=['name'], ascending=[True])
            df.to_parquet(dst_parquet_file, index=False)

            with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
                print('---', file=f)
                print('license: apache-2.0', file=f)
                print('language:', file=f)
                print('- en', file=f)
                print('tags:', file=f)
                print('- python', file=f)
                print('- code', file=f)
                print('- downloads', file=f)
                print('- pypistats', file=f)
                print('size_categories:', file=f)
                print(f'- {number_to_tag(len(df))}', file=f)
                print('source_datasets:', file=f)
                print('- pypistats', file=f)
                print('---', file=f)
                print('', file=f)

                # Description
                print('# PyPI Download Statistics Dataset', file=f)
                print('', file=f)
                print('This dataset contains download statistics for Python packages from PyPI (Python Package Index). '
                      'The data is collected from pypistats and includes recent download counts for packages, '
                      'providing insights into package popularity and usage trends.', file=f)
                print('', file=f)

                # Dataset overview
                total_rows = len(df)
                updated_rows = len(df[df['updated_at'].notna()])
                non_empty_rows = len(df[(~df['is_empty']) & df['updated_at'].notna()])

                print('## Dataset Overview', file=f)
                print('', file=f)
                print(f'- **Total packages**: {total_rows:,}', file=f)
                print(f'- **Packages with data**: {updated_rows:,} ({updated_rows / total_rows * 100:.1f}%)', file=f)
                print(f'- **Non-empty packages**: {non_empty_rows:,} ({non_empty_rows / total_rows * 100:.1f}%)',
                      file=f)
                print('', file=f)

                # Schema description
                print('## Schema', file=f)
                print('', file=f)
                print('| Column | Type | Description |', file=f)
                print('|--------|------|-------------|', file=f)
                print('| name | string | Package name on PyPI |', file=f)
                print('| url | string | PyPI package URL |', file=f)
                print('| last_day | integer | Downloads in the last day |', file=f)
                print('| last_week | integer | Downloads in the last week |', file=f)
                print('| last_month | integer | Downloads in the last month |', file=f)
                print('| is_empty | boolean | Whether the package has no download data |', file=f)
                print('| updated_at | float | Unix timestamp of last update |', file=f)
                print('', file=f)

                # Sample data
                print('## Sample Data', file=f)
                print('', file=f)
                sample_df = df.head(20)[['name', 'last_day', 'last_week', 'last_month', 'is_empty']]
                print('First 20 packages:', file=f)
                print('', file=f)
                print(sample_df.to_markdown(index=False), file=f)
                print('', file=f)

                # Top packages by downloads
                if non_empty_rows > 0:
                    top_df = df[(~df['is_empty']) & df['updated_at'].notna()].nlargest(20, 'last_month')[
                        ['name', 'last_day', 'last_week', 'last_month']]
                    print('## Top 20 Packages by Monthly Downloads', file=f)
                    print('', file=f)
                    print(top_df.to_markdown(index=False), file=f)
                    print('', file=f)

                    # Statistics
                    stats_df = df[(~df['is_empty']) & df['updated_at'].notna()]
                    if len(stats_df) > 0:
                        print('## Download Statistics', file=f)
                        print('', file=f)
                        print('### Monthly Downloads', file=f)
                        print(f'- **Total**: {stats_df["last_month"].sum():,}', file=f)
                        print(f'- **Average**: {stats_df["last_month"].mean():.0f}', file=f)
                        print(f'- **Median**: {stats_df["last_month"].median():.0f}', file=f)
                        print(f'- **Max**: {stats_df["last_month"].max():,}', file=f)
                        print('', file=f)

                        print('### Weekly Downloads', file=f)
                        print(f'- **Total**: {stats_df["last_week"].sum():,}', file=f)
                        print(f'- **Average**: {stats_df["last_week"].mean():.0f}', file=f)
                        print(f'- **Median**: {stats_df["last_week"].median():.0f}', file=f)
                        print(f'- **Max**: {stats_df["last_week"].max():,}', file=f)
                        print('', file=f)

                        print('### Daily Downloads', file=f)
                        print(f'- **Total**: {stats_df["last_day"].sum():,}', file=f)
                        print(f'- **Average**: {stats_df["last_day"].mean():.0f}', file=f)
                        print(f'- **Median**: {stats_df["last_day"].median():.0f}', file=f)
                        print(f'- **Max**: {stats_df["last_day"].max():,}', file=f)
                        print('', file=f)

            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=upload_dir,
                path_in_repo='.',
                message=f'Update PyPI - {total_rows:,} packages, {non_empty_rows:,} with data',
                clear=True,
            )

        has_update = False
        last_saved_at = time.time()

    def _make_item(pypi_name: str):
        nonlocal has_update
        updated_at = time.time()
        data = get_pypistats_recent(pypi_name, session=session)
        if not data:
            logging.warning(f'No data found for {pypi_name!r}, skipped.')
            d_records[pypi_name]['last_day'] = None
            d_records[pypi_name]['last_week'] = None
            d_records[pypi_name]['last_month'] = None
            d_records[pypi_name]['is_empty'] = True
        else:
            d_records[pypi_name]['last_day'] = data['data']['last_day']
            d_records[pypi_name]['last_week'] = data['data']['last_week']
            d_records[pypi_name]['last_month'] = data['data']['last_month']
            d_records[pypi_name]['is_empty'] = False
        d_records[pypi_name]['updated_at'] = updated_at

        with lock:
            has_update = True
            _deploy(force=False)

    parallel_call(
        iterable=natsorted(df_x['name']),
        fn=_make_item,
        desc='Getting data'
    )

    _deploy(force=True)


if __name__ == '__main__':
    # Set up colored logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    sync(
        repository='HansBug/pypi_downloads',
        proxy_pool=os.environ['PP_URL']
    )
