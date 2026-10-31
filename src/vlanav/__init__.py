"""VLA-NAV: reliability-aware replanning for pretrained navigation VLAs.

Modules:
    environment : simulators + episode benchmark definitions
    model       : NavPolicy contract + mock for tests; OmniVLA adapter (Kaggle)
    reliability : progress signal, trigger thresholds, replanning regimes
    evaluation  : metrics, statistics, logging
    demo        : trajectory/first-person rendering and video export
"""
__all__ = ["environment", "model", "reliability", "evaluation", "demo"]
__version__ = "0.1.0"
