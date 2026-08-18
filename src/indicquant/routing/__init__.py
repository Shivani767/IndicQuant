"""Router instrumentation — the RQ4 surface.

`SarvamMoEGate.forward` returns `(topk_idx, topk_weight, logits)`, so one forward hook per
MoE layer yields every quantity RQ4 needs. See ARCHITECTURE.md §1.3.
"""

from indicquant.routing.hooks import RouterRecorder, RoutingCaptureError
from indicquant.routing.metrics import (
    RoutingComparisonError,
    expert_activation_histogram,
    expert_coverage,
    language_affinity_matrix,
    router_entropy,
    topk_agreement,
)

__all__ = [
    "RouterRecorder",
    "RoutingCaptureError",
    "RoutingComparisonError",
    "expert_activation_histogram",
    "expert_coverage",
    "language_affinity_matrix",
    "router_entropy",
    "topk_agreement",
]
