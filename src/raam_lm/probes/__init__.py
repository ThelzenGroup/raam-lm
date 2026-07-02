"""Deterministic synthetic probes."""

from .copy_task import make_copy_batch
from .assoc_recall import make_assoc_recall_batch
from .passkey import make_passkey_batch
from .state_tracking import make_state_tracking_batch

PROBE_BUILDERS = {
    "copy": make_copy_batch,
    "assoc_recall": make_assoc_recall_batch,
    "passkey": make_passkey_batch,
    "state_tracking": make_state_tracking_batch,
}

__all__ = ["PROBE_BUILDERS"]

