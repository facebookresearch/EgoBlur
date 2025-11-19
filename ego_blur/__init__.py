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
Public package interface for EgoBlur demos.

This package re-exports the most common classes used by downstream code and
provides a stable module name for the Python package that will be distributed
on PyPI.
"""

from gen2.script import ClassID, EgoblurDetector, main as gen2_main

__all__ = ["ClassID", "EgoblurDetector", "gen2_main"]
