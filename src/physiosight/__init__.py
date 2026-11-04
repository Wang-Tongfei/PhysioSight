"""PhysioSight - single-camera, real-time, markerless movement analysis for rehabilitation exercises.

Not a medical device: outputs are exercise guidance and research data, not diagnoses.
"""
from importlib.metadata import PackageNotFoundError, version

NAME = __name__  # the distribution, import and command name are all "physiosight" (pyproject.toml)
try:
    __version__ = version(NAME)
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"
