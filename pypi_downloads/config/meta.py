"""
Metadata definitions for the :mod:`pypi_downloads` package.

This module defines the core metadata constants used by package configuration
tools such as ``setup.py`` or build scripts. The values describe the project
identity, versioning, and author contact information.

The module contains the following main components:

* :data:`__TITLE__` - Project title
* :data:`__VERSION__` - Project version string
* :data:`__DESCRIPTION__` - Short project description
* :data:`__AUTHOR__` - Project author name
* :data:`__AUTHOR_EMAIL__` - Author contact email

Example::

    >>> from pypi_downloads.config import meta
    >>> meta.__TITLE__
    'pypi_downloads'
    >>> meta.__VERSION__
    '0.0.1'

"""

#: Title of this project (should be `pypi_downloads`).
__TITLE__: str = 'pypi_downloads'

#: Version of this project.
__VERSION__: str = '0.0.1'

#: Short description of the project, will be included in ``setup.py``.
__DESCRIPTION__: str = 'Offline data for pypi downloads'

#: Author of this project.
__AUTHOR__: str = 'HansBug'

#: Email of the authors'.
__AUTHOR_EMAIL__: str = 'hansbug@buaa.edu.cn'
