from .scorer import GeneralizationReport, RegressionAlert, compare, run
from .suite import BUILTIN_PAIRS, NearMissPair, get_suite

__all__ = [
    "BUILTIN_PAIRS",
    "GeneralizationReport",
    "NearMissPair",
    "RegressionAlert",
    "compare",
    "get_suite",
    "run",
]

