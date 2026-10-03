"""ORBIT: function-space preconditioning for rotary Query/Key updates."""

from .adapter import OrbitAdapter, QKPairSpec
from .baselines import Muon
from .model import OrbitGPT, OrbitGPTConfig
from .optimizer import Orbit
from .paper_baselines import (
    PAPER_BASELINE_NAMES,
    AdaMuon,
    NorMuon,
    PaperAdamW,
    PublishedAdaMuon,
    build_paper_baseline,
)

__version__ = "0.1.0"

__all__ = [
    "Orbit",
    "Muon",
    "PaperAdamW",
    "NorMuon",
    "AdaMuon",
    "PublishedAdaMuon",
    "build_paper_baseline",
    "PAPER_BASELINE_NAMES",
    "OrbitGPT",
    "OrbitGPTConfig",
    "OrbitAdapter",
    "QKPairSpec",
    "__version__",
]
