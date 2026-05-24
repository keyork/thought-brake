"""thought-brake: early stopping for reasoning models."""

from .client import ThoughtBrakeClient
from .config import EarlyStopConfig
from .detectors import BudgetDetector, CompressionDetector, NoStopDetector, StopDecision
from .types import ChatResponse, RequestMetrics, StopReason

__all__ = [
    "ThoughtBrakeClient",
    "EarlyStopConfig",
    "BudgetDetector",
    "CompressionDetector",
    "NoStopDetector",
    "StopDecision",
    "ChatResponse",
    "RequestMetrics",
    "StopReason",
]
