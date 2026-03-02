"""
Update version number in pypi_downloads/config/meta.py to current date.

This script updates the __VERSION__ constant in meta.py to the current date
in the format YYYY.M.D (e.g., 2026.7.18), while preserving all other content
in the file exactly as-is.

Usage:
    python -m tools.update_version
"""

import re
from datetime import datetime
from pathlib import Path


def update_version():
    """Update the version number in meta.py to current date (YYYY.M.D format)."""
    # Get the project root directory
    meta_file = Path('pypi_downloads') / 'config' / 'meta.py'

    if not meta_file.exists():
        raise FileNotFoundError(f"meta.py not found at {meta_file}")

    # Generate new version number based on current date
    now = datetime.now()
    new_version = f"{now.year}.{now.month}.{now.day}"

    # Read the current content
    with open(meta_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace only the __VERSION__ line using regex
    # This pattern matches the __VERSION__ assignment line while preserving formatting
    pattern = r"(__VERSION__\s*:\s*str\s*=\s*)['\"]([^'\"]+)['\"]"

    def replace_version(match):
        prefix = match.group(1)
        old_version = match.group(2)
        quote = "'" if "'" in match.group(0) else "'"
        print(f"Updating version: {old_version} -> {new_version}")
        return f"{prefix}'{new_version}'"

    new_content = re.sub(pattern, replace_version, content)

    # Write back to file
    with open(meta_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"✓ Version updated to {new_version} in {meta_file}")
    return new_version


if __name__ == '__main__':
    update_version()
