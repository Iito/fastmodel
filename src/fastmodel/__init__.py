__version__ = "0.0.0.dev0"  # Replaced by CI from git tag at build time

from .protocol import ServingModule

__all__ = ["__version__", "ServingModule"]
