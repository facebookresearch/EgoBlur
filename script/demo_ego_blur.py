# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from ego_blur import ImageAnonymizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--face_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur face model file path",
    )

    parser.add_argument(
        "--face_model_score_threshold",
        required=False,
        type=float,
        default=0.9,
        help=(
            "Face model score threshold to filter out low confidence detections"
        ),
    )

    parser.add_argument(
        "--lp_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur license plate model file path",
    )

    parser.add_argument(
        "--lp_model_score_threshold",
        required=False,
        type=float,
        default=0.9,
        help=(
            "License plate model score threshold to filter out low confidence"
            " detections"
        ),
    )

    parser.add_argument(
        "--nms_iou_threshold",
        required=False,
        type=float,
        default=0.3,
        help="NMS iou threshold to filter out low confidence overlapping boxes",
    )

    parser.add_argument(
        "--scale_factor_detections",
        required=False,
        type=float,
        default=1,
        help=(
            "Scale detections by the given factor to allow blurring more area,"
            " 1.15 would mean 15% scaling"
        ),
    )

    parser.add_argument(
        "--input_image_path",
        required=False,
        type=str,
        default=None,
        help=(
            "Absolute path for the given image on which we want to make"
            " detections"
        ),
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
        help=(
            "Absolute path for the given video on which we want to make"
            " detections"
        ),
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


def create_output_directory(file_path: str) -> None:
    """
    parameter file_path: absolute path to the directory where we want to create the output files
    Simple logic to create output directories if they don't exist.
    """
    print(
        f"Directory {os.path.dirname(file_path)} does not exist. Attempting to"
        " create it..."
    )
    os.makedirs(os.path.dirname(file_path))
    if not os.path.exists(os.path.dirname(file_path)):
        raise ValueError(
            f"Directory {os.path.dirname(file_path)} didn't exist. Attempt to"
            " create also failed. Please provide another path."
        )


def validate_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """
    parameter args: parsed arguments
    Run some basic checks on the input arguments
    """
    # input args value checks
    if not 0.0 <= args.face_model_score_threshold <= 1.0:
        raise ValueError(
            "Invalid face_model_score_threshold"
            f" {args.face_model_score_threshold}"
        )
    if not 0.0 <= args.lp_model_score_threshold <= 1.0:
        raise ValueError(
            f"Invalid lp_model_score_threshold {args.lp_model_score_threshold}"
        )
    if not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise ValueError(f"Invalid nms_iou_threshold {args.nms_iou_threshold}")
    if not 0 <= args.scale_factor_detections:
        raise ValueError(
            f"Invalid scale_factor_detections {args.scale_factor_detections}"
        )
    if not 1 <= args.output_video_fps or not (
        isinstance(args.output_video_fps, int)
        and args.output_video_fps % 1 == 0
    ):
        raise ValueError(
            f"Invalid output_video_fps {args.output_video_fps}, should be a"
            " positive integer"
        )

    # input/output paths checks
    if args.face_model_path is None and args.lp_model_path is None:
        raise ValueError(
            "Please provide either face_model_path or lp_model_path or both"
        )
    if args.input_image_path is None and args.input_video_path is None:
        raise ValueError(
            "Please provide either input_image_path or input_video_path"
        )
    if args.input_image_path is not None and args.output_image_path is None:
        raise ValueError(
            "Please provide output_image_path for the visualized image to save."
        )
    if args.input_video_path is not None and args.output_video_path is None:
        raise ValueError(
            "Please provide output_video_path for the visualized video to save."
        )
    if args.input_image_path is not None and not os.path.exists(
        args.input_image_path
    ):
        raise ValueError(f"{args.input_image_path} does not exist.")
    if args.input_video_path is not None and not os.path.exists(
        args.input_video_path
    ):
        raise ValueError(f"{args.input_video_path} does not exist.")
    if args.face_model_path is not None and not os.path.exists(
        args.face_model_path
    ):
        raise ValueError(f"{args.face_model_path} does not exist.")
    if args.lp_model_path is not None and not os.path.exists(
        args.lp_model_path
    ):
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
            f"You don't have permissions to write to {args.output_image_path}."
            " Please grant adequate permissions, or provide a different output"
            " path."
        )
    if args.output_video_path is not None and not os.access(
        os.path.dirname(args.output_video_path), os.W_OK
    ):
        raise ValueError(
            f"You don't have permissions to write to {args.output_video_path}."
            " Please grant adequate permissions, or provide a different output"
            " path."
        )

    return args


def read_image(image_path: str) -> np.ndarray:
    """
    parameter image_path: absolute path to an image
    Return an image in BGR format
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


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    args = validate_inputs(parse_args())

    image_anonymizer = ImageAnonymizer(
        args.face_model_path,
        args.lp_model_path,
        args.face_model_score_threshold,
        args.lp_model_score_threshold,
        args.nms_iou_threshold,
        args.scale_factor_detections,
    )

    if args.input_image_path is not None:
        bgr_image = read_image(args.input_image_path)

        image = image_anonymizer.visualize_image(bgr_image=bgr_image)

        write_image(image=image, image_path=args.output_image_path)

    if args.input_video_path is not None:
        visualized_images = []
        video_reader_clip = VideoFileClip(args.input_video_path)

        for frame in video_reader_clip.iter_frames():
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            visualized_images.append(
                image_anonymizer.visualize_image(bgr_image=frame)
            )

        video_reader_clip.close()

        if visualized_images:
            video_writer_clip = ImageSequenceClip(
                visualized_images, fps=args.output_video_fps
            )
            video_writer_clip.write_videofile(args.output_video_path)
            video_writer_clip.close()
