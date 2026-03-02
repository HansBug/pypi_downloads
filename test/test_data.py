import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from pypi_downloads.data import (
    _DATA_FILE, _HF_FILENAME, _HF_REPO,
    _ensure_data_file, _freeze_dataframe, _load_cached, load_data,
)


def _make_sample_df():
    return pd.DataFrame({
        'name': ['numpy', 'pandas', 'requests'],
        'last_day': pd.array([1000, 2000, 300], dtype='int64'),
        'last_week': pd.array([7000, 14000, 2100], dtype='int64'),
        'last_month': pd.array([30000, 60000, 9000], dtype='int64'),
        'updated_at': [1.0e9, 1.1e9, 1.2e9],
    })


@pytest.fixture(autouse=True)
def clear_load_cache():
    _load_cached.cache_clear()
    yield
    _load_cached.cache_clear()


@pytest.fixture
def sample_parquet(tmp_path):
    path = str(tmp_path / 'downloads.parquet')
    _make_sample_df().to_parquet(path, index=False)
    return path


@pytest.fixture
def full_hf_parquet(tmp_path):
    """A parquet mimicking the unfiltered HuggingFace dataset.parquet."""
    df = pd.DataFrame({
        'name': ['numpy', 'flask', 'broken-pkg'],
        'url': ['u1', 'u2', 'u3'],
        'last_day': [100.0, 200.0, None],
        'last_week': [700.0, 1400.0, None],
        'last_month': [3000.0, 6000.0, None],
        'status': ['valid', 'valid', 'empty'],
        'updated_at': [1e9, 1e9, None],
    })
    path = str(tmp_path / 'dataset.parquet')
    df.to_parquet(path, index=False)
    return path


@pytest.mark.unittest
class TestEnsureDataFile:
    def test_file_exists_returns_immediately(self):
        with patch('pypi_downloads.data.os.path.exists', return_value=True):
            _ensure_data_file()  # must not raise or touch HF Hub

    def test_missing_no_hf_hub(self):
        with patch('pypi_downloads.data.os.path.exists', return_value=False):
            with patch.dict(sys.modules, {'huggingface_hub': None}):
                with pytest.raises(FileNotFoundError) as exc_info:
                    _ensure_data_file()
        msg = str(exc_info.value)
        assert 'not installed' in msg
        assert 'make download_data' in msg
        assert 'huggingface_hub' in msg

    def test_missing_download_fails(self):
        mock_hf = MagicMock()
        mock_hf.hf_hub_download.side_effect = RuntimeError('network error')
        with patch('pypi_downloads.data.os.path.exists', return_value=False):
            with patch.dict(sys.modules, {'huggingface_hub': mock_hf}):
                with pytest.raises(FileNotFoundError) as exc_info:
                    _ensure_data_file()
        msg = str(exc_info.value)
        assert 'failed' in msg
        assert 'make download_data' in msg
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_missing_download_succeeds(self, tmp_path, full_hf_parquet):
        dst = str(tmp_path / 'downloads.parquet')
        mock_hf = MagicMock()
        mock_hf.hf_hub_download.return_value = full_hf_parquet

        with patch('pypi_downloads.data.os.path.exists', return_value=False):
            with patch.dict(sys.modules, {'huggingface_hub': mock_hf}):
                with patch('pypi_downloads.data._DATA_FILE', dst):
                    _ensure_data_file()

        # Assertions outside patches so os.path.exists is unpatched
        assert os.path.exists(dst)
        result = pd.read_parquet(dst)
        assert list(result.columns) == ['name', 'last_day', 'last_week', 'last_month', 'updated_at']
        assert set(result['name'].tolist()) == {'numpy', 'flask'}  # 'empty' filtered out
        assert result['last_day'].dtype == np.int64
        assert result['last_week'].dtype == np.int64
        assert result['last_month'].dtype == np.int64
        assert result['updated_at'].dtype == np.float64
        mock_hf.hf_hub_download.assert_called_once_with(
            repo_id=_HF_REPO, repo_type='dataset', filename=_HF_FILENAME
        )


@pytest.mark.unittest
class TestFreezeDataFrame:
    def test_returns_same_instance(self):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        result = _freeze_dataframe(df)
        assert result is df

    def test_numpy_object_column_frozen(self):
        # pandas 2.1+ with future.infer_string may store strings as ArrowStringArray
        # (already immutable). Only assert flags when backing is actually numpy.
        df = pd.DataFrame({'name': np.array(['numpy', 'pandas'], dtype=object)})
        _freeze_dataframe(df)
        values = df['name'].values
        if isinstance(values, np.ndarray):
            assert not values.flags.writeable

    def test_numpy_float_column_frozen(self):
        df = pd.DataFrame({'val': [1.0, 2.0, 3.0]})
        _freeze_dataframe(df)
        assert not df['val'].values.flags.writeable

    def test_numpy_mutation_raises(self):
        df = pd.DataFrame({'val': [1.0, 2.0]})
        _freeze_dataframe(df)
        with pytest.raises(ValueError):
            df['val'].values[0] = 99.0

    def test_extension_array_data_frozen(self):
        df = pd.DataFrame({'count': pd.array([1, 2, 3], dtype='Int64')})
        _freeze_dataframe(df)
        col_arr = df['count'].array
        assert not col_arr._data.flags.writeable

    def test_extension_array_mask_frozen(self):
        df = pd.DataFrame({'count': pd.array([1, 2, 3], dtype='Int64')})
        _freeze_dataframe(df)
        col_arr = df['count'].array
        assert not col_arr._mask.flags.writeable

    def test_mixed_dtypes(self):
        df = pd.DataFrame({
            'name': np.array(['numpy', 'pandas'], dtype=object),
            'count': pd.array([1, 2], dtype='Int64'),
            'ratio': [0.5, 1.5],
        })
        _freeze_dataframe(df)
        name_vals = df['name'].values
        if isinstance(name_vals, np.ndarray):  # numpy-backed; Arrow is already immutable
            assert not name_vals.flags.writeable
        assert not df['count'].array._data.flags.writeable
        assert not df['ratio'].values.flags.writeable

    def test_empty_dataframe(self):
        df = pd.DataFrame({'a': pd.Series([], dtype='float64')})
        result = _freeze_dataframe(df)
        assert result is df


@pytest.mark.unittest
class TestLoadCached:
    def test_returns_dataframe(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df = _load_cached()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['name', 'last_day', 'last_week', 'last_month', 'updated_at']
        assert len(df) == 3

    def test_caching_returns_same_instance(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df1 = _load_cached()
            df2 = _load_cached()
        assert df1 is df2

    def test_result_is_frozen(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df = _load_cached()
        # int64 is always numpy-backed regardless of pandas/pyarrow version
        assert not df['last_day'].values.flags.writeable

    def test_data_values_correct(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df = _load_cached()
        assert df['name'].tolist() == ['numpy', 'pandas', 'requests']
        assert df['last_day'].tolist() == [1000, 2000, 300]

    def test_missing_file_raises(self, tmp_path):
        missing = str(tmp_path / 'no_such_file.parquet')
        with patch('pypi_downloads.data._DATA_FILE', missing):
            with patch.dict(sys.modules, {'huggingface_hub': None}):
                with pytest.raises(FileNotFoundError):
                    _load_cached()


@pytest.mark.unittest
class TestLoadData:
    def test_returns_dataframe(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df = load_data()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['name', 'last_day', 'last_week', 'last_month', 'updated_at']

    def test_same_object_as_load_cached(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df1 = load_data()
            df2 = _load_cached()
        assert df1 is df2

    def test_result_is_frozen(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df = load_data()
        # int64 is always numpy-backed regardless of pandas/pyarrow version
        assert not df['last_day'].values.flags.writeable

    def test_caching_across_calls(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df1 = load_data()
            df2 = load_data()
        assert df1 is df2

    def test_writable_false_returns_same_object(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df1 = load_data(writable=False)
            df2 = load_data(writable=False)
        assert df1 is df2

    def test_writable_true_returns_copy(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df_frozen = load_data(writable=False)
            df_copy = load_data(writable=True)
        assert df_copy is not df_frozen

    def test_writable_true_is_mutable(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df = load_data(writable=True)
        df.loc[0, 'last_day'] = 99  # must not raise

    def test_writable_true_does_not_affect_cache(self, sample_parquet):
        with patch('pypi_downloads.data._DATA_FILE', sample_parquet):
            df_copy = load_data(writable=True)
            df_copy.loc[0, 'last_day'] = 99
            df_frozen = load_data(writable=False)
        assert df_frozen['last_day'].tolist()[0] == 1000
