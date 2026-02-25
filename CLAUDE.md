# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pypi_downloads` is a tool for collecting and syncing offline PyPI download statistics. It fetches download counts (last day/week/month) for all packages on PyPI via the pypistats.org API and stores them as a Parquet dataset on HuggingFace Hub.

## Commands

### Testing
```bash
# Run all unit tests
make unittest

# Run a single test file
UNITTEST=1 pytest test/config/test_meta.py -sv -m unittest

# Run tests with coverage report
make unittest COV_TYPES="xml term-missing"

# Run tests in parallel
make unittest WORKERS=4

# Run tests in a specific subdirectory
make unittest RANGE_DIR=config
```

### Building & Packaging
```bash
make package   # Build sdist and wheel into dist/
make clean     # Remove dist/, build/, *.egg-info
```

### Docs
```bash
make docs      # Build docs
make pdocs     # Build production docs
make rst_auto  # Auto-generate RST files from Python source
make docs_auto # Regenerate docs via LLM (uses remake_docs_via_llm.py)
```

### Running Tools Directly
```bash
# Sync PyPI download data to HuggingFace dataset
python tools/sync.py   # Requires PP_URL env var for proxy pool

# Fetch PyPI package index
python tools/pypi.py

# Fetch recent stats for a specific package
python tools/pypistats/recent.py
```

## Architecture

### Package (`pypi_downloads/`)
The main installable package — currently minimal (just `config/meta.py` with version/author metadata). The CLI entry point is `pypi_downloads.entry:pypi_downloadscli`.

### Tools (`tools/`)
Scripts not part of the installable package, used for data collection and syncing:

- **`tools/sync.py`** — Core sync logic. Fetches the full PyPI index, queries pypistats for download stats on stale packages (not updated in 30+ days), and uploads the result as `dataset.parquet` to a HuggingFace dataset repo. Uses `parallel_call` for concurrent requests. Periodically saves intermediate results (every `deploy_span` seconds, default 5 min). Also generates distribution charts (PNG) and a README for the HF dataset.
- **`tools/pypi.py`** — Fetches and parses the PyPI simple index (`https://pypi.org/simple`) into `PypiItem` dataclass objects.
- **`tools/pypistats/recent.py`** — Queries `https://pypistats.org/api/packages/{name}/recent` for last-day/week/month download counts.
- **`tools/utils/session.py`** — Shared HTTP session factory with retry logic, timeout handling, and random user-agent rotation.

### Data Flow
1. `get_pypi_index()` → list of all PyPI packages
2. Load existing dataset from HF Hub (`dataset.parquet` or `dataset.csv`) if it exists
3. Filter to records needing refresh (never updated, or updated >30 days ago)
4. `parallel_call` → `get_pypistats_recent()` for each stale package
5. Periodically upload updated Parquet + README + charts to HF Hub dataset

### Test Structure (`test/`)
Tests mirror the `pypi_downloads/` package structure. Test classes use `@pytest.mark.unittest`. The `make unittest` target sets `UNITTEST=1` env var.

## Key Dependencies
- **`hfutils`** — HuggingFace upload/download utilities
- **`hbutils`** — Utilities including `parallel_call` for concurrent processing and `ColoredFormatter` for logging
- **`pandas`/`numpy`** — Data manipulation and statistics
- **`matplotlib`** — Chart generation for the HF dataset README
- **`pyquery`** / **`requests`** — HTML parsing and HTTP requests
- **`random_user_agent`** — Rotating user agents to avoid rate limiting
