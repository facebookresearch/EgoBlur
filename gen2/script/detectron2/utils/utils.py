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

import sys
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from gen2.script.detectron2.structures import Instances
from PIL import Image


def convert_scripted_instances(instances):
    """
    Convert a scripted Instances object to a regular Instances object.
    """

    assert hasattr(
        instances, "image_size"
    ), f"Expect an Instances object, but got {type(instances)}!"
    ret = Instances(instances.image_size)
    for name in instances._field_names:
        val = getattr(instances, "_" + name, None)
        if val is not None:
            ret.set(name, val)
    return ret


def detector_postprocess(
    results: Instances,
    output_height: int,
    output_width: int,
) -> Instances:
    """
    Simplified copy of detectron2.modeling.postprocessing.detector_postprocess
    without mask/keypoint handling.
    """

    scale_x = output_width / results.image_size[1]
    scale_y = output_height / results.image_size[0]
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    return results


class ResizeTransform:
    """
    Minimal resize transform compatible with TorchScriptPredictor.
    """

    def __init__(
        self, h: int, w: int, new_h: int, new_w: int, interp: int = Image.BILINEAR
    ):
        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w
        self.interp = interp

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] != (self.h, self.w):
            raise ValueError(
                f"Unexpected input shape {img.shape[:2]}, expected {(self.h, self.w)}"
            )

        if img.dtype == np.uint8:
            if img.ndim == 3 and img.shape[2] == 1:
                pil_img = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_img = Image.fromarray(img)
            resized = pil_img.resize((self.new_w, self.new_h), self.interp)
            arr = np.asarray(resized)
            if img.ndim == 3 and img.shape[2] == 1:
                arr = np.expand_dims(arr, -1)
            return arr

        # fallback to torch interpolate for floating types
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        shape = list(tensor.shape)
        shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
        tensor = tensor.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
        mode = {
            Image.NEAREST: "nearest",
            Image.BILINEAR: "bilinear",
            Image.BICUBIC: "bicubic",
        }[self.interp]
        align_corners = None if mode == "nearest" else False
        tensor = F.interpolate(
            tensor, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
        )
        shape[:2] = [self.new_h, self.new_w]
        return tensor.permute(2, 3, 0, 1).view(shape).numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords = coords.copy()
        coords[:, 0] = coords[:, 0] * (self.new_w / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h / self.h)
        return coords


class ResizeShortestEdge:
    """
    Minimal ResizeShortestEdge augmentation that mirrors detectron2 functionality.
    """

    def __init__(
        self,
        short_edge_length: Union[int, Tuple[int, int], List[int]],
        max_size: int = sys.maxsize,
        sample_style: str = "range",
        interp: int = Image.BILINEAR,
    ) -> None:
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self.short_edge_length = tuple(short_edge_length)
        self.max_size = max_size
        if sample_style not in {"range", "choice"}:
            raise ValueError(f"Unsupported sample_style '{sample_style}'")
        self.sample_style = sample_style
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> ResizeTransform:
        h, w = image.shape[:2]
        if self.sample_style == "range":
            low, high = self.short_edge_length
            size = np.random.randint(low, high + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        size = max(int(size), 1)

        new_h, new_w = self.get_output_shape(h, w, size, self.max_size)
        return ResizeTransform(h, w, new_h, new_w, self.interp)

    @staticmethod
    def get_output_shape(
        oldh: int, oldw: int, short_edge_length: int, max_size: int
    ) -> Tuple[int, int]:
        size = float(short_edge_length)
        h, w = oldh, oldw
        scale = size / min(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size
        if max(new_h, new_w) > max_size:
            scale = max_size / max(new_h, new_w)
            new_h *= scale
            new_w *= scale
        return int(new_h + 0.5), int(new_w + 0.5)
