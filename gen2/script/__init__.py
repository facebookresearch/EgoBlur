# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""
Expose Gen2 scripts and helpers as a package.

During packaging we vendor a minimal subset of Detectron2 utilities under
``gen2.script.detectron2``.  TorchScript models serialized with the original
Detectron2 still reference modules under the top-level ``detectron2`` package
namespace.  To keep the wheel self‑contained we create lightweight aliases
pointing to the vendored modules when the real Detectron2 package is absent.
"""

import sys
from types import ModuleType

from gen2.script import detectron2 as _vendored_detectron2
from gen2.script.demo_ego_blur_gen2 import main
from gen2.script.detectron2 import (
    export as _vendored_export,
    structures as _vendored_structures,
    utils as _vendored_utils,
)
from gen2.script.detectron2.structures import (
    boxes as _vendored_boxes,
    instances as _vendored_instances,
)
from gen2.script.predictor import ClassID, EgoblurDetector

__all__ = ["ClassID", "EgoblurDetector", "main"]


def _alias_vendored_detectron2() -> None:
    """Register vendored detectron2 modules under the original namespace."""
    if "detectron2" in sys.modules:
        # Respect a real detectron2 installation when present.
        return

    detectron2_module = ModuleType("detectron2")
    detectron2_module.__dict__.update(
        {
            "__file__": getattr(_vendored_detectron2, "__file__", None),
            "__package__": "detectron2",
            "__doc__": getattr(_vendored_detectron2, "__doc__", None),
            "__path__": [],  # namespace package semantics
            "__version__": getattr(_vendored_detectron2, "__version__", None),
            "structures": _vendored_structures,
            "utils": _vendored_utils,
            "export": _vendored_export,
        }
    )

    alias_map = {
        "detectron2": detectron2_module,
        "detectron2.structures": _vendored_structures,
        "detectron2.utils": _vendored_utils,
        "detectron2.export": _vendored_export,
        "detectron2.structures.boxes": _vendored_boxes,
        "detectron2.structures.instances": _vendored_instances,
    }

    for name, module in alias_map.items():
        if hasattr(module, "__name__"):
            module.__name__ = name
        sys.modules[name] = module

    if hasattr(_vendored_boxes, "Boxes"):
        _vendored_boxes.Boxes.__module__ = "detectron2.structures.boxes"
    if hasattr(_vendored_instances, "Instances"):
        _vendored_instances.Instances.__module__ = "detectron2.structures.instances"


_alias_vendored_detectron2()
