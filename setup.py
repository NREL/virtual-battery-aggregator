import io
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = [
    # 'ochre',
    'numpy',
    'pandas',
    'cvxpy',  # Note: use conda with channel conda-forge (conda install -c conda-forge cvxpy)
]


# Read the version from the __init__.py file without importing it
def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name='aggregator',
      version=find_version('aggregator', '__init__.py'),
      description='An aggregator for distributed batteries using OCHRE battery models for the FAST-DERMS project',
      author='Michael Blonsky',
      author_email='Michael.Blonsky@nrel.gov',
      url='https://github.com/NREL/virtual-battery-aggregator',
      packages=['aggregator'],
      install_requires=requirements,
      )
