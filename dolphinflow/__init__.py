# dolphinfow/__init__.py

# The standard, dependency-free version is always available.
from .dolphinflow import DolphinFlow

# The 8-bit version is an optional feature. We try to import it,
# but if the dependency is not met, we don't expose it and the
# package remains usable.
try:
    from .dolphinflow_8bit import DolphinFlow8bit
except ImportError:
    # If bitsandbytes is not installed, DolphinFlow8bit will not be available.
    # We pass silently, as this is expected behavior for an optional dependency.
    pass

__all__ = ['DolphinFlow', 'DolphinFlow8bit']
__version__ = '0.2.0'
