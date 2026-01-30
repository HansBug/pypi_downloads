import logging
import os.path
import time
from threading import Lock
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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
        has_status = 'status' in df.columns
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
        has_status = 'status' in df.columns
    else:
        logging.info(f'No existing file found.')
        names = set()
        records = []
        has_status = False

    for name, item in d_index.items():
        if name not in names:
            records.append({
                'name': name,
                'url': item.url,
                'last_day': None,
                'last_month': None,
                'last_week': None,
                'status': None,
                'updated_at': None,
            })

    d_records = {item['name']: item for item in records}

    df_x = pd.DataFrame(records)
    if not has_status:
        df_x['status'] = None
        df_x.loc[~df_x['updated_at'].isnull(), 'status'] = 'empty'
        df_x.loc[~df_x['last_month'].isnull(), 'status'] = 'valid'

    d_records = {item['name']: item for item in df_x.replace(np.nan, None).to_dict('records')}
    df_x = df_x[
        df_x['updated_at'].isnull() |
        (~df_x['updated_at'].isnull() & (df_x['updated_at'] + 30 * 86400 < time.time()))
        ]
    logging.info(f'Records to refresh:\n{df_x}')

    has_update = False
    last_saved_at = None
    lock = Lock()

    def _generate_charts(df_non_empty, upload_dir):
        """Generate distribution charts for download statistics"""
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'

        # Set up the figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PyPI Package Download Distribution Analysis', fontsize=16, fontweight='bold')

        periods = ['last_day', 'last_week', 'last_month']
        period_labels = ['Daily Downloads', 'Weekly Downloads', 'Monthly Downloads']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, (period, label, color) in enumerate(zip(periods, period_labels, colors)):
            data = df_non_empty[period].dropna()
            data = data[data > 0]  # Remove zero values for log scale

            if len(data) == 0:
                continue

            # Log-scale histogram (top row)
            ax_hist = axes[0, i]
            log_data = np.log10(data)
            ax_hist.hist(log_data, bins=50, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
            ax_hist.set_xlabel('Log₁₀(Downloads)')
            ax_hist.set_ylabel('Number of Packages')
            ax_hist.set_title(f'{label} Distribution (Log Scale)')
            ax_hist.grid(True, alpha=0.3)

            # Add percentile annotations
            percentiles = [50, 90, 95, 99]
            for p in percentiles:
                pct_val = np.percentile(data, p)
                ax_hist.axvline(np.log10(pct_val), color='red', linestyle='--', alpha=0.7)
                ax_hist.text(np.log10(pct_val), ax_hist.get_ylim()[1] * 0.8,
                             f'P{p}\n{pct_val:,.0f}',
                             rotation=90, ha='right', va='top', fontsize=8)

            # Cumulative distribution (bottom row)
            ax_cum = axes[1, i]
            sorted_data = np.sort(data)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
            ax_cum.semilogx(sorted_data, cumulative, color=color, linewidth=2)
            ax_cum.set_xlabel('Downloads (Log Scale)')
            ax_cum.set_ylabel('Cumulative Percentage (%)')
            ax_cum.set_title(f'{label} Cumulative Distribution')
            ax_cum.grid(True, alpha=0.3)
            ax_cum.set_xlim(left=1)

            # Add key statistics as text
            stats_text = f'Total: {data.sum():,}\nMean: {data.mean():.0f}\nMedian: {data.median():.0f}\nMax: {data.max():,}'
            ax_cum.text(0.02, 0.98, stats_text, transform=ax_cum.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=9)

        plt.tight_layout()
        chart_path = os.path.join(upload_dir, 'download_distribution.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Generate a separate chart for top packages comparison
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Get top 20 packages by monthly downloads
        top_packages = df_non_empty.nlargest(20, 'last_month')

        x = np.arange(len(top_packages))
        width = 0.25

        bars1 = ax.bar(x - width, top_packages['last_day'], width, label='Daily', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x, top_packages['last_week'], width, label='Weekly', color='#ff7f0e', alpha=0.8)
        bars3 = ax.bar(x + width, top_packages['last_month'], width, label='Monthly', color='#2ca02c', alpha=0.8)

        ax.set_xlabel('Package Name')
        ax.set_ylabel('Downloads (Log Scale)')
        ax.set_title('Top 20 PyPI Packages by Download Volume')
        ax.set_xticks(x)
        ax.set_xticklabels(top_packages['name'], rotation=45, ha='right')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Add value labels on bars (only for monthly downloads to avoid clutter)
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom', fontsize=8, rotation=90)

        plt.tight_layout()
        top_packages_path = os.path.join(upload_dir, 'top_packages.png')
        plt.savefig(top_packages_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return ['download_distribution.png', 'top_packages.png']

    def _deploy(force=False):
        nonlocal has_update, last_saved_at
        if not has_update:
            return
        if not force and last_saved_at is not None and last_saved_at + deploy_span > time.time():
            return

        with TemporaryDirectory() as upload_dir:
            dst_parquet_file = os.path.join(upload_dir, 'dataset.parquet')
            logging.info(f'Saving to {dst_parquet_file}')
            df = pd.DataFrame(list(d_records.values()))
            df = df.sort_values(by=['name'], ascending=[True])
            df.to_parquet(dst_parquet_file, index=False)

            # Dataset overview
            total_rows = len(df)
            df_notna = df[df['updated_at'].notna()]
            with_data_rows = len(df_notna)
            df_non_empty = df_notna[df_notna['status'] == 'valid']
            non_empty_rows = len(df_non_empty)

            # Generate charts
            chart_files = []
            if non_empty_rows > 0:
                logging.info('Generating distribution charts...')
                chart_files = _generate_charts(df_non_empty, upload_dir)

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

                print('## Dataset Overview', file=f)
                print('', file=f)
                print(f'- **Total packages**: {total_rows:,}', file=f)
                print(f'- **Packages with data**: {with_data_rows:,} ({with_data_rows / total_rows * 100:.1f}%)',
                      file=f)
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
                print('| status | string | Whether the package has no download data (null, empty, valid) |', file=f)
                print('| updated_at | float | Unix timestamp of last update |', file=f)
                print('', file=f)

                # Distribution Analysis Charts
                if chart_files and non_empty_rows > 0:
                    print('## Download Distribution Analysis', file=f)
                    print('', file=f)
                    print('The following charts show the distribution of download statistics across all PyPI packages. '
                          'As expected, the data exhibits a long-tail distribution where a small number of packages '
                          'receive the majority of downloads.', file=f)
                    print('', file=f)

                    if 'download_distribution.png' in chart_files:
                        print('### Distribution Overview', file=f)
                        print('', file=f)
                        print('![Download Distribution](download_distribution.png)', file=f)
                        print('', file=f)
                        print('**Top row**: Histogram showing the distribution of downloads on a logarithmic scale. '
                              'The red dashed lines indicate key percentiles (P50, P90, P95, P99).', file=f)
                        print('', file=f)
                        print('**Bottom row**: Cumulative distribution showing what percentage of packages '
                              'have downloads below a given threshold.', file=f)
                        print('', file=f)

                    if 'top_packages.png' in chart_files:
                        print('### Top Packages Comparison', file=f)
                        print('', file=f)
                        print('![Top Packages](top_packages.png)', file=f)
                        print('', file=f)
                        print(
                            'Comparison of daily, weekly, and monthly download volumes for the top 20 most downloaded packages.',
                            file=f)
                        print('', file=f)

                # Sample data
                print('## Sample Data', file=f)
                print('', file=f)
                sample_df = df_non_empty.head(20)[['name', 'last_day', 'last_week', 'last_month', 'status']]
                print('First 20 packages:', file=f)
                print('', file=f)
                print(sample_df.to_markdown(index=False), file=f)
                print('', file=f)

                # Top packages by downloads
                if non_empty_rows > 0:
                    top_df = df_non_empty.nlargest(20, 'last_month')[
                        ['name', 'last_day', 'last_week', 'last_month']]
                    print('## Top 20 Packages by Monthly Downloads', file=f)
                    print('', file=f)
                    print(top_df.to_markdown(index=False), file=f)
                    print('', file=f)

                    # Statistics
                    stats_df = df_non_empty
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
                message=f'Update PyPI - {total_rows:,} packages, '
                        f'{with_data_rows:,} ({with_data_rows / total_rows * 100:.1f}%) with data, '
                        f'{non_empty_rows:,} ({non_empty_rows / total_rows * 100:.1f}%) non empty',
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
            d_records[pypi_name]['status'] = 'empty'
        else:
            d_records[pypi_name]['last_day'] = data['data']['last_day']
            d_records[pypi_name]['last_week'] = data['data']['last_week']
            d_records[pypi_name]['last_month'] = data['data']['last_month']
            d_records[pypi_name]['status'] = 'valid'
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
