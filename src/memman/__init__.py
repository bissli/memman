from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('memman')
except PackageNotFoundError:
    __version__ = 'dev'
