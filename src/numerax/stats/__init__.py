"""Statistics submodule for numerax."""

from . import chi2
from ._profile import make_profile_llh

__all__ = ["chi2", "make_profile_llh"]
