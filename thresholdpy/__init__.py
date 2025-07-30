"""
ThresholdPy: A Python adaptation of ThresholdR for CITE-seq denoising with ScanPy integration
"""

from .thresholdpy import ThresholdPy, pp_threshold_proteins

__version__ = "0.1.3"
__author__ = "Ross Jones"
__email__ = "jonesr18@gmail.com"

__all__ = [
    "ThresholdPy",
    "pp_threshold_proteins",
]
