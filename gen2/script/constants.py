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

"""
Gen2-specific constants shared by demo scripts and consumers of the EgoBlur package.

These values are exposed so downstream users can import the same resize configuration
and detection thresholds after installing the package via pip.
"""

from typing import Dict

RESIZE_MIN_GEN2: int = 1200
RESIZE_MAX_GEN2: int = 1200

# Score threshold bounds shared with Gen2 predictor validation
MIN_MODEL_SCORE_THRESHOLD_GEN2: float = 0.0
MAX_MODEL_SCORE_THRESHOLD_GEN2: float = 1.0

# Camera-specific default thresholds for face detection
FACE_THRESHOLDS_GEN2: Dict[str, float] = {
    "slam-front-left": 0.69862,
    "slam-front-right": 0.69862,
    "slam-side-left": 0.68160,
    "slam-side-right": 0.68160,
    "camera-rgb": 0.67416,
}

# Camera-specific default thresholds for license plate detection
LP_THRESHOLDS_GEN2: Dict[str, float] = {
    "slam-front-left": 0.95245,
    "slam-front-right": 0.95245,
    "slam-side-left": 0.95242,
    "slam-side-right": 0.95242,
    "camera-rgb": 0.74475,
}

__all__ = [
    "RESIZE_MIN_GEN2",
    "RESIZE_MAX_GEN2",
    "MIN_MODEL_SCORE_THRESHOLD_GEN2",
    "MAX_MODEL_SCORE_THRESHOLD_GEN2",
    "FACE_THRESHOLDS_GEN2",
    "LP_THRESHOLDS_GEN2",
]
