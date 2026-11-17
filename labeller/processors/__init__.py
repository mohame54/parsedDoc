from __future__ import annotations

from .base import DatasetProcessor, NativeLabels
from .bilingual_rx import BilingualRxProcessor
from .funsd import FunsdProcessor
from .generic import GenericProcessor
from .rxhandbd import RxHandBDProcessor
from .sidecar import SidecarTextProcessor
from .sroie import SroieProcessor

# DocILE is routed to the generic vision processor until its real files are inspected.
PROCESSORS: dict[str, type[DatasetProcessor]] = {
    "funsd": FunsdProcessor,
    "sroie": SroieProcessor,
    "rxhandbd": RxHandBDProcessor,
    "bilingual_rx": BilingualRxProcessor,
    "sidecar_text": SidecarTextProcessor,
    "generic": GenericProcessor,
    "docile": GenericProcessor,
}


def get_processor(name: str) -> DatasetProcessor:
    if name not in PROCESSORS:
        raise ValueError(f"unknown processor {name!r}; known: {sorted(PROCESSORS)}")
    return PROCESSORS[name]()


__all__ = ["DatasetProcessor", "NativeLabels", "get_processor", "PROCESSORS"]
