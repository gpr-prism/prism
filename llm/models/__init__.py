"""
Models package for PRISM and baselines.
"""

from .common import ModelConfig, CONFIGS, CausalLM, LowRankLinear, count_parameters
from .gdn import GDNBlock
from .efla import EFLABlock
from .pgdn import PGDNBlock
from .prism import PRISMBlock
from .prism_ablations import PRISMr2Block, PRISMr3Block, PRISMr4Block, PRISMr5Block

# Registry: model name -> block class
MODEL_REGISTRY = {
    "gdn":      GDNBlock,
    "efla":     EFLABlock,
    "pgdn":     PGDNBlock,
    "prism":    PRISMBlock,
    "prism_r2": PRISMr2Block,
    "prism_r3": PRISMr3Block,
    "prism_r4": PRISMr4Block,
    "prism_r5": PRISMr5Block,
}

__all__ = [
    "ModelConfig", "CONFIGS", "CausalLM", "LowRankLinear", "count_parameters",
    "GDNBlock", "EFLABlock", "PGDNBlock", "PRISMBlock",
    "PRISMr2Block", "PRISMr3Block", "PRISMr4Block", "PRISMr5Block",
    "MODEL_REGISTRY",
]
