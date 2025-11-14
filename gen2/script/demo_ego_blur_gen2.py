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

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from typing import List, Optional

import cv2
import numpy as np
from detectron2.export.torchscript_patch import patch_instances
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from predictor import ClassID, EgoblurDetector, PATCH_INSTANCES_FIELDS

from tqdm.auto import tqdm
from utils import (
    get_device,
    get_image_tensor,
    read_image,
    scale_box,
    setup_logger,
    validate_inputs,
    write_image,
)


logger = setup_logger()

# Default resize configuration fed into `EgoblurDetector` so demo scripts continue
# to resize frames to 1200 on the short edge while capping the long edge at 1200.
RESIZE_MIN = 1200
RESIZE_MAX = 1200


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--face_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur Gen2 face model file path",
    )

    parser.add_argument(
        "--face_model_score_threshold",
        required=False,
        type=float,
        default=0.05,
        help="Face model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--lp_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur Gen2 license plate model file path",
    )

    parser.add_argument(
        "--lp_model_score_threshold",
        required=False,
        type=float,
        default=0.05,
        help="License plate model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--nms_iou_threshold",
        required=False,
        type=float,
        default=0.5,
        help="NMS iou threshold to filter out low confidence overlapping boxes",
    )

    parser.add_argument(
        "--scale_factor_detections",
        required=False,
        type=float,
        default=1,
        help="Scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling",
    )

    parser.add_argument(
        "--input_image_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path for the given image on which we want to make detections",
    )

    parser.add_argument(
        "--output_image_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path where we want to store the visualized image",
    )

    parser.add_argument(
        "--input_video_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path for the given video on which we want to make detections",
    )

    parser.add_argument(
        "--output_video_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path where we want to store the visualized video",
    )

    parser.add_argument(
        "--output_video_fps",
        required=False,
        type=int,
        default=30,
        help="FPS for the output video",
    )

    return parser.parse_args()


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
        w = x2 - x1
        h = y2 - y1

        ksize = (image.shape[0] // 2, image.shape[1] // 2)
        image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)
        cv2.ellipse(mask, (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 0), 255, -1)

    inverse_mask = cv2.bitwise_not(mask)
    image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
    image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
    image = cv2.add(image_bg, image_fg)

    return image


def visualize_image(
    input_image_path: str,
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    output_image_path: str,
    scale_factor_detections: float,
):
    """
    parameter input_image_path: absolute path to the input image
    parameter face_detector: face detector helper (may be None)
    parameter lp_detector: license plate detector helper (may be None)
    parameter output_image_path: absolute path where the visualized image will be saved
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area

    Perform detections on the input image and save the output image at the given path.
    """
    bgr_image = read_image(input_image_path)
    image = bgr_image.copy()

    image_tensor = get_image_tensor(bgr_image)
    detections = []

    with patch_instances(fields=PATCH_INSTANCES_FIELDS):
        # get face detections
        if face_detector is not None:
            face_results = face_detector.run(image_tensor)
            if face_results:
                if len(face_results) != 1:
                    raise ValueError(
                        f"EgoblurDetector.run is expected to return results for a single "
                        f"image in this script, got {len(face_results)}."
                    )
                detections.extend(face_results[0])

        # get license plate detections
        if lp_detector is not None:
            lp_results = lp_detector.run(image_tensor)
            if lp_results:
                if len(lp_results) > 1:
                    raise ValueError(
                        f"EgoblurDetector.run is expected to return results for a single "
                        f"image in this script, got {len(lp_results)}."
                    )
                detections.extend(lp_results[0])

    image = visualize(
        image,
        detections,
        scale_factor_detections,
    )
    write_image(image, output_image_path)


def visualize_video(
    input_video_path: str,
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    output_video_path: str,
    scale_factor_detections: float,
    output_video_fps: int,
) -> None:
    """
    parameter input_video_path: absolute path to the input video
    parameter face_detector: face detector helper (may be None)
    parameter lp_detector: license plate detector helper (may be None)
    parameter output_video_path: absolute path where the visualized video will be saved
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area
    parameter output_video_fps: FPS for the output video

    Perform detections on the input video and save the output video at the given path.
    """
    visualized_frames: List[np.ndarray] = []
    video_reader_clip = VideoFileClip(input_video_path)

    try:
        with patch_instances(fields=PATCH_INSTANCES_FIELDS):
            frame_iterator = video_reader_clip.iter_frames()
            total_frames: Optional[int] = None
            reader = getattr(video_reader_clip, "reader", None)
            if reader is not None:
                nframes = getattr(reader, "nframes", None)
                if (
                    isinstance(nframes, (int, float))
                    and math.isfinite(nframes)
                    and nframes > 0
                ):
                    total_frames = int(nframes)
            if total_frames is None:
                duration = getattr(video_reader_clip, "duration", None)
                fps = getattr(video_reader_clip, "fps", None)
                if isinstance(duration, (int, float)) and isinstance(fps, (int, float)):
                    estimated_total = duration * fps
                    if math.isfinite(estimated_total) and estimated_total > 0:
                        total_frames = int(estimated_total)

            progress_iterator = frame_iterator
            if tqdm is not None:
                progress_iterator = tqdm(
                    frame_iterator,
                    total=total_frames,
                    desc="Processing frames",
                    unit="frame",
                )

            try:
                for frame in progress_iterator:
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                    bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    image_tensor = get_image_tensor(bgr_image)
                    detections: List[List[float]] = []

                    if face_detector is not None:
                        face_results = face_detector.run(image_tensor)
                        if len(face_results) != 1:
                            raise ValueError(
                                "EgoblurDetector.run is expected to return results "
                                f"for a single image in this script, got {len(face_results)}."
                            )
                        detections.extend(face_results[0])
                    if lp_detector is not None:
                        lp_results = lp_detector.run(image_tensor)
                        if len(lp_results) != 1:
                            raise ValueError(
                                "EgoblurDetector.run is expected to return results "
                                f"for a single image in this script, got {len(lp_results)}."
                            )
                        detections.extend(lp_results[0])

                    visualized_bgr = visualize(
                        bgr_image.copy(),
                        detections,
                        scale_factor_detections,
                    )
                    if visualized_bgr.dtype != np.uint8:
                        visualized_bgr = np.clip(visualized_bgr, 0, 255).astype(
                            np.uint8
                        )

                    visualized_rgb = cv2.cvtColor(visualized_bgr, cv2.COLOR_BGR2RGB)
                    visualized_frames.append(np.ascontiguousarray(visualized_rgb))
            finally:
                if tqdm is not None and hasattr(progress_iterator, "close"):
                    progress_iterator.close()
    finally:
        video_reader_clip.close()

    if not visualized_frames:
        raise ValueError(
            f"No frames were processed from {input_video_path}. "
            "Please verify the input video file."
        )

    clip = ImageSequenceClip(visualized_frames, fps=output_video_fps)
    try:
        clip.write_videofile(
            output_video_path,
            codec="libx264",
            audio=False,
            fps=output_video_fps,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        logger.info(f"Successfully output video to:{output_video_path}")
    finally:
        clip.close()


if __name__ == "__main__":
    args = validate_inputs(parse_args())
    device = get_device()

    face_detector: Optional[EgoblurDetector]
    if args.face_model_path is not None:
        face_detector = EgoblurDetector(
            model_path=args.face_model_path,
            device=device,
            detection_class=ClassID.FACE,
            score_threshold=args.face_model_score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            resize_aug={
                "min_size_test": RESIZE_MIN,
                "max_size_test": RESIZE_MAX,
            },
        )
    else:
        face_detector = None

    lp_detector: Optional[EgoblurDetector]
    if args.lp_model_path is not None:
        lp_detector = EgoblurDetector(
            model_path=args.lp_model_path,
            device=device,
            detection_class=ClassID.LICENSE_PLATE,
            score_threshold=args.lp_model_score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            resize_aug={
                "min_size_test": RESIZE_MIN,
                "max_size_test": RESIZE_MAX,
            },
        )
    else:
        lp_detector = None

    if args.input_image_path is not None:
        visualize_image(
            args.input_image_path,
            face_detector,
            lp_detector,
            args.output_image_path,
            args.scale_factor_detections,
        )

    if args.input_video_path is not None:
        visualize_video(
            args.input_video_path,
            face_detector,
            lp_detector,
            args.output_video_path,
            args.scale_factor_detections,
            args.output_video_fps,
        )
