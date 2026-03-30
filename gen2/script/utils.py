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

import argparse
import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def create_output_directory(file_path: str) -> None:
    """
    parameter file_path: absolute path to the directory where we want to create the output files
    Simple logic to create output directories if they don't exist.
    """
    print(
        f"Directory {os.path.dirname(file_path)} does not exist. Attempting to create it..."
    )
    os.makedirs(os.path.dirname(file_path))
    if not os.path.exists(os.path.dirname(file_path)):
        raise ValueError(
            f"Directory {os.path.dirname(file_path)} didn't exist. Attempt to create also failed. Please provide another path."
        )


def validate_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """
    parameter args: parsed arguments
    Run some basic checks on the input arguments
    """
    # input args value checks
    if args.face_model_score_threshold is not None and not (
        0.0 <= args.face_model_score_threshold <= 1.0
    ):
        raise ValueError(
            f"Invalid face_model_score_threshold {args.face_model_score_threshold}"
        )
    if args.lp_model_score_threshold is not None and not (
        0.0 <= args.lp_model_score_threshold <= 1.0
    ):
        raise ValueError(
            f"Invalid lp_model_score_threshold {args.lp_model_score_threshold}"
        )
    if not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise ValueError(f"Invalid nms_iou_threshold {args.nms_iou_threshold}")
    if not 0 <= args.scale_factor_detections:
        raise ValueError(
            f"Invalid scale_factor_detections {args.scale_factor_detections}"
        )

    # input/output paths checks
    if args.face_model_path is None and args.lp_model_path is None:
        raise ValueError(
            "Please provide either face_model_path or lp_model_path or both"
        )
    if args.input_image_path is None and args.input_video_path is None:
        raise ValueError("Please provide either input_image_path or input_video_path")
    if args.input_image_path is not None and args.output_image_path is None:
        raise ValueError(
            "Please provide output_image_path for the visualized image to save."
        )
    if args.input_video_path is not None and args.output_video_path is None:
        raise ValueError(
            "Please provide output_video_path for the visualized video to save."
        )
    if args.input_image_path is not None and not os.path.exists(args.input_image_path):
        raise ValueError(f"{args.input_image_path} does not exist.")
    if args.input_video_path is not None and not os.path.exists(args.input_video_path):
        raise ValueError(f"{args.input_video_path} does not exist.")
    if args.face_model_path is not None and not os.path.exists(args.face_model_path):
        raise ValueError(f"{args.face_model_path} does not exist.")
    if args.lp_model_path is not None and not os.path.exists(args.lp_model_path):
        raise ValueError(f"{args.lp_model_path} does not exist.")
    if args.output_image_path is not None and not os.path.exists(
        os.path.dirname(args.output_image_path)
    ):
        create_output_directory(args.output_image_path)
    if args.output_video_path is not None and not os.path.exists(
        os.path.dirname(args.output_video_path)
    ):
        create_output_directory(args.output_video_path)

    # check we have write permissions on output paths
    if args.output_image_path is not None and not os.access(
        os.path.dirname(args.output_image_path), os.W_OK
    ):
        raise ValueError(
            f"You don't have permissions to write to {args.output_image_path}. Please grant adequate permissions, or provide a different output path."
        )
    if args.output_video_path is not None and not os.access(
        os.path.dirname(args.output_video_path), os.W_OK
    ):
        raise ValueError(
            f"You don't have permissions to write to {args.output_video_path}. Please grant adequate permissions, or provide a different output path."
        )

    return args


@lru_cache
def get_device() -> str:
    """
    Return the device type
    """
    return (
        "cpu"
        if not torch.cuda.is_available()
        else f"cuda:{torch.cuda.current_device()}"
    )


def read_image(image_path: str) -> np.ndarray:
    """
    parameter image_path: absolute path to an image
    Return an image in BGR format, CHW
    """
    bgr_image = cv2.imread(image_path)
    if len(bgr_image.shape) == 2:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    return bgr_image


def write_image(image: np.ndarray, image_path: str) -> None:
    """
    parameter image: np.ndarray in BGR format
    parameter image_path: absolute path where we want to save the visualized image
    """
    cv2.imwrite(image_path, image)
    logger.info(f"Successfully write image to: {image_path}")


def get_image_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    """
    parameter bgr_image: image on which we want to make detections

    Return the image tensor in CHW format (BGR order)
    """
    bgr_image_transposed = np.transpose(bgr_image, (2, 0, 1))
    image_tensor = torch.from_numpy(bgr_image_transposed).to(get_device())

    return image_tensor


def scale_box(
    box: List[List[float]], max_width: int, max_height: int, scale: float
) -> List[List[float]]:
    """
    parameter box: detection box in format (x1, y1, x2, y2)
    parameter scale: scaling factor

    Returns a scaled bbox as (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1
    h = y2 - y1

    xc = x1 + w / 2
    yc = y1 + h / 2

    w = scale * w
    h = scale * h

    x1 = max(xc - w / 2, 0)
    y1 = max(yc - h / 2, 0)

    x2 = min(xc + w / 2, max_width)
    y2 = min(yc + h / 2, max_height)

    return [x1, y1, x2, y2]


def _get_threshold(
    camera_name: Optional[str],
    user_threshold: Optional[float],
    threshold_map: Optional[Dict[str, float]],
) -> float:
    """
    Resolve the effective score threshold using user input or camera defaults.
    """
    if user_threshold is not None:
        return user_threshold
    if threshold_map is not None:
        if camera_name is not None:
            return threshold_map.get(camera_name, threshold_map["camera-rgb"])
        else:
            return threshold_map["camera-rgb"]
    raise ValueError(
        "Cannot retrieve the model score threshold. Please provide a user-specified threshold or a mapping from camera name to threshold."
    )


def visualize(
    image: np.ndarray,
    detections: List[List[float]],
    scale_factor_detections: float,
) -> np.ndarray:
    """
    parameter image: image on which we want to make detections
    parameter detections: list of bounding boxes in format [x1, y1, x2, y2]
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling

    Visualize the input image with the detections and save the output image at the given path
    """
    image_fg = image.copy()
    mask_shape = (image.shape[0], image.shape[1], 1)
    mask = np.full(mask_shape, 0, dtype=np.uint8)

    for box in detections:
        if scale_factor_detections != 1.0:
            box = scale_box(
                box, image.shape[1], image.shape[0], scale_factor_detections
            )
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # Clamp to image bounds to avoid empty slices.
        img_h, img_w = image.shape[:2]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        ksize = (image.shape[0] // 2, image.shape[1] // 2)
        image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)
        cv2.ellipse(mask, (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 0), 255, -1)

    inverse_mask = cv2.bitwise_not(mask)
    image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
    image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
    image = cv2.add(image_bg, image_fg)

    return image
