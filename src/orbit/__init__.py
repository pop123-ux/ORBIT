"""ORBIT research optimizer package.

ORBIT is an optimizer-only research path for RoPE attention. The model remains a
standard Transformer at inference; training additionally accumulates tiny 2x2
query/key covariance statistics for every RoPE frequency pair.
"""

from .model import OrbitGPT, OrbitGPTConfig
from .optimizer import Orbit

__all__ = ["Orbit", "OrbitGPT", "OrbitGPTConfig"]
