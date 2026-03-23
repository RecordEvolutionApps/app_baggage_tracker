"""Abstract base class for inference engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import supervision as sv


class InferenceEngine(ABC):
    """Contract that every inference backend must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (e.g. ``'huggingface'``, ``'mmdet'``)."""

    @abstractmethod
    def available(self) -> bool:
        """Return ``True`` when this engine can run on the current platform."""

    @abstractmethod
    def load_model(self, model_name: str, config=None) -> Dict[str, Any]:
        """Load a model by name, returning a *model_bundle* dict.

        The bundle must include at least ``{'backend': self.name, 'model_name': ...}``.
        Engine-specific keys (``model``, ``processor``, ``inferencer``, …) are private
        to the engine and only consumed by :meth:`infer`.
        """

    @abstractmethod
    def infer(self, frame: np.ndarray, model_bundle: Dict[str, Any],
              confidence: float | None = None, iou: float | None = None,
              config=None) -> Optional[sv.Detections]:
        """Run inference on a single frame, returning ``sv.Detections`` or ``False``."""

    @abstractmethod
    def list_models(self) -> Dict[str, Any]:
        """Return the model zoo dict for this engine."""

    def supports_model(self, model_name: str) -> bool:
        """Check whether *model_name* is in this engine's zoo."""
        return model_name in self.list_models()
