import logging
import os.path
import time
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd
from hbutils.concurrent import parallel_call
from hbutils.logging import ColoredFormatter
from natsort import natsorted

from .pypi import get_pypi_index
from .pypistats.recent import get_pypistats_recent
from .utils import get_requests_session


def sync(dst_file: str, proxy_pool: Optional[str] = None, deploy_span: float = 5 * 60):
    session = get_requests_session()
    if proxy_pool:
        logging.info(f'Proxy pool {proxy_pool!r} enabled.')
        session.proxies.update({
            'http': proxy_pool,
            'https': proxy_pool
        })

    logging.info('Getting index ...')
    d_index = {item.name: item for item in get_pypi_index(session=session)}

    if os.path.exists(dst_file):
        logging.info(f'Load from {dst_file!r} ...')
        df = pd.read_csv(dst_file).replace(np.nan, None)
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
                'updated_at': None,
            })

    d_records = {item['name']: item for item in records}

    df_x = pd.DataFrame(records)
    df_x = df_x[
        df_x['last_month'].isnull() |
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

        logging.info(f'Saving to {dst_file}')
        df = pd.DataFrame(list(d_records.values()))
        df = df.sort_values(by=['name'], ascending=[True])
        df.to_csv(dst_file, index=False)
        has_update = False
        last_saved_at = time.time()

    def _make_item(pypi_name: str):
        nonlocal has_update
        updated_at = time.time()
        data = get_pypistats_recent(pypi_name, session=session)
        if not data:
            logging.warning(f'No data found for {pypi_name!r}, skipped.')
            return

        d_records[pypi_name]['last_day'] = data['data']['last_day']
        d_records[pypi_name]['last_week'] = data['data']['last_week']
        d_records[pypi_name]['last_month'] = data['data']['last_month']
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
        dst_file='pypi_downloads/data.csv',
        proxy_pool=os.environ['PP_URL']
    )
