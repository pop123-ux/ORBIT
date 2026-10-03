"""ORBIT: function-space preconditioning for rotary Query/Key updates."""

from .adapter import OrbitAdapter, QKPairSpec
from .baselines import Muon
from .model import OrbitGPT, OrbitGPTConfig
from .optimizer import Orbit

__version__ = "0.1.0"

__all__ = [
    "Orbit",
    "Muon",
    "OrbitGPT",
    "OrbitGPTConfig",
    "OrbitAdapter",
    "QKPairSpec",
    "__version__",
]
