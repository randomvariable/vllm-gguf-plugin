# SPDX-License-Identifier: Apache-2.0

# Patch gguf enum for ROCmFPX custom types before any GGUF reading
from . import rocmfpx_types  # noqa: F401

# Patch gguf enum for ik_llama.cpp K-variant i-quant types
from . import ik_types  # noqa: F401

from .config_parser import GGUFConfigParser
from .loader import GGUFModelLoader
from .plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from .quantization import DiffusionGGUFConfig, GGUFConfig

__all__ = [
    "DiffusionGGUFConfig",
    "GGUFConfig",
    "GGUFConfigParser",
    "GGUFModelLoader",
    "OOTGGUFConfig",
    "OOTGGUFModelLoader",
    "register",
]
