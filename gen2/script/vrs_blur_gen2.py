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
End-to-end VRS blur tool using Gen2 EgoBlur models.

Reads camera streams from a VRS file, runs face and/or license-plate detection
(via egoblur Gen2), applies blur, H.265-encodes the blurred frames (via PyAV),
and writes everything back into a single output VRS file.  Non-camera streams
(IMU, VIO, barometer, eye tracking, etc.) are copied verbatim by the underlying
VRS AsyncImageFilter infrastructure.

All dependencies are OSS:
  - pyvrs (``pip install vrs``) for VRS reading/writing via AsyncImageFilter
  - egoblur (this package) for detection + blur
  - av (``pip install av``) for H.265 video encoding/decoding
"""

import argparse
import fractions
import heapq
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import torch
from gen2.script.constants import (
    FACE_THRESHOLDS_GEN2,
    LP_THRESHOLDS_GEN2,
    RESIZE_MAX_GEN2,
    RESIZE_MIN_GEN2,
)
from gen2.script.detectron2.export.torchscript_patch import patch_instances
from gen2.script.predictor import ClassID, EgoblurDetector, PATCH_INSTANCES_FIELDS
from gen2.script.utils import (
    _get_threshold,
    get_device,
    get_image_tensor,
    setup_logger,
    visualize,
)
from pyvrs import (
    FilteredFileReader,
    ImageBuffer,
    OssAsyncImageFilter as AsyncImageFilter,
)

logger = setup_logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CAMERA_STREAM_IDS: List[str] = [
    "214-1",  # RGB camera
    "1201-1",  # SLAM camera 1
    "1201-2",  # SLAM camera 2
    "1201-3",  # SLAM camera 3
    "1201-4",  # SLAM camera 4
]

RGB_STREAM_PREFIX = "214"


def _is_rgb_stream(stream_id: str) -> bool:
    return stream_id.startswith(RGB_STREAM_PREFIX)


class FrameTimings:
    """Accumulates per-phase timing stats for a single stream."""

    def __init__(self) -> None:
        self.decode_ms: List[float] = []
        self.inference_ms: List[float] = []
        self.blur_ms: List[float] = []
        self.encode_ms: List[float] = []

    def summary(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for name, vals in [
            ("decode", self.decode_ms),
            ("inference", self.inference_ms),
            ("blur", self.blur_ms),
            ("encode", self.encode_ms),
        ]:
            if vals:
                s = sorted(vals)
                n = len(s)
                result[name] = {
                    "mean": sum(s) / n,
                    "p50": s[n // 2],
                    "p95": s[int(n * 0.95)],
                    "total_s": sum(s) / 1000.0,
                }
        return result


class H265Decoder:
    """Decode H.265 byte buffers to numpy images using PyAV/libx265."""

    def __init__(self, is_rgb: bool) -> None:
        self._codec = av.CodecContext.create("hevc", "r")
        self._codec.open()
        self._is_rgb = is_rgb

    def decode_frame(self, data: bytes) -> Optional[np.ndarray]:
        """Decode a single H.265 packet and return an RGB or grayscale numpy array."""
        packet = av.Packet(data)
        try:
            frames = self._codec.decode(packet)
        except av.error.InvalidDataError:
            return None
        for frame in frames:
            if self._is_rgb:
                return frame.to_ndarray(format="rgb24")
            else:
                return frame.to_ndarray(format="gray")
        return None


# ---------------------------------------------------------------------------
# PyAV H.265 encoder
# ---------------------------------------------------------------------------


def _nvenc_available() -> bool:
    """Return True if hevc_nvenc (NVIDIA GPU encoder) is usable."""
    try:
        probe = av.CodecContext.create("hevc_nvenc", "w")
        probe.width = 64
        probe.height = 64
        probe.pix_fmt = "yuv420p"
        probe.time_base = fractions.Fraction(1, 30)
        probe.open()
        probe.close()
        return True
    except Exception:
        return False


_USE_NVENC: Optional[bool] = None


def _should_use_nvenc() -> bool:
    global _USE_NVENC
    if _USE_NVENC is None:
        _USE_NVENC = _nvenc_available()
        if _USE_NVENC:
            logger.info(
                "GPU encoder (hevc_nvenc) detected — using CUDA for H.265 encoding"
            )
        else:
            logger.info("GPU encoder not available — falling back to CPU libx265")
    return _USE_NVENC


class H265Encoder:
    """Encode individual frames to H.265 byte buffers.

    Automatically uses hevc_nvenc (NVIDIA GPU) when available, otherwise
    falls back to libx265 (CPU).
    """

    def __init__(
        self,
        width: int,
        height: int,
        is_rgb: bool,
        quality: int = 18,
        preset: str = "ultrafast",
    ) -> None:
        use_gpu = _should_use_nvenc()
        if use_gpu and not is_rgb:
            use_gpu = False

        encoder_name = "hevc_nvenc" if use_gpu else "libx265"
        codec = av.CodecContext.create(encoder_name, "w")
        codec.width = width
        codec.height = height
        codec.pix_fmt = "yuv420p" if is_rgb else "gray"
        codec.time_base = fractions.Fraction(1, 30)
        codec.gop_size = 1

        if use_gpu:
            codec.options = {
                "preset": "p1",
                "rc": "constqp",
                "qp": str(quality),
            }
        else:
            codec.options = {
                "preset": preset,
                "crf": str(quality),
                "x265-params": "log-level=warning:bframes=0:max-num-reorder-pics=0:rc-lookahead=0:frame-threads=1:lookahead-slices=0",
            }
        codec.open()
        self._codec = codec
        self._pts = 0
        self._encoder_name = encoder_name

    def encode_frame(self, image: np.ndarray) -> bytes:
        """Encode a single numpy image and return the raw H.265 bytes."""
        frame = av.VideoFrame.from_ndarray(image, format=self._codec.pix_fmt)
        frame.pts = self._pts
        self._pts += 1

        encoded = b""
        for packet in self._codec.encode(frame):
            encoded += bytes(packet)
        return encoded

    def flush(self) -> bytes:
        """Flush any remaining packets from the encoder."""
        encoded = b""
        for packet in self._codec.encode():
            encoded += bytes(packet)
        return encoded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blur faces / license plates in VRS camera streams (OSS-only)."
    )

    parser.add_argument(
        "--input_vrs_path",
        type=str,
        required=True,
        help="Path to the input VRS file.",
    )
    parser.add_argument(
        "--output_vrs_path",
        type=str,
        required=True,
        help="Path for the output blurred VRS file.",
    )
    parser.add_argument(
        "--stream_ids",
        type=str,
        nargs="+",
        default=DEFAULT_CAMERA_STREAM_IDS,
        help="VRS stream IDs to process (default: Aria RGB + SLAM cameras).",
    )

    parser.add_argument(
        "--face_model_path",
        type=str,
        default=None,
        help="Path to the Gen2 face TorchScript model.",
    )
    parser.add_argument(
        "--lp_model_path",
        type=str,
        default=None,
        help="Path to the Gen2 license-plate TorchScript model.",
    )

    parser.add_argument(
        "--camera_name",
        type=str,
        default=None,
        choices=[
            "slam-front-left",
            "slam-front-right",
            "slam-side-left",
            "slam-side-right",
            "camera-rgb",
        ],
        help="Camera name for camera-specific default thresholds.",
    )
    parser.add_argument("--face_model_score_threshold", type=float, default=None)
    parser.add_argument("--lp_model_score_threshold", type=float, default=None)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    parser.add_argument("--scale_factor_detections", type=float, default=1.0)

    parser.add_argument(
        "--encoding_quality",
        type=int,
        default=18,
        help="H.265 CRF value (0=lossless, 51=worst). Default 18.",
    )
    parser.add_argument(
        "--encoding_preset",
        type=str,
        default="ultrafast",
        help="x265 preset (ultrafast, superfast, veryfast, faster, fast, "
        "medium, slow, slower, veryslow).",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of frames to batch together for model inference. "
        "Higher values improve GPU throughput but use more memory. Default 1.",
    )

    parser.add_argument(
        "--debug_encoding_only",
        action="store_true",
        default=False,
        help="Skip detection/blur — just decode and re-encode frames. "
        "Useful for testing the VRS write path without model inference.",
    )

    parser.add_argument(
        "--aug_using_cpu",
        action="store_true",
        default=False,
        help="Use CPU-based PIL resize (ResizeShortestEdge) instead of "
        "GPU-based F.interpolate for pre-processing. Slower but matches "
        "the original detectron2 augmentation pipeline exactly.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Batched detection + blur
# ---------------------------------------------------------------------------


def _detect_and_blur_batch(
    images_bgr: List[np.ndarray],
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    scale_factor: float,
) -> Tuple[List[np.ndarray], float, float]:
    """Run batched face/LP detection and blur on a list of BGR images.

    Returns (blurred_images, inference_ms, blur_ms).
    """
    n = len(images_bgr)
    if n == 0:
        return [], 0.0, 0.0

    image_tensors = [get_image_tensor(img) for img in images_bgr]
    batched_tensor = torch.stack(image_tensors)

    all_detections: List[List[List[float]]] = [[] for _ in range(n)]

    t_inf = time.time()
    if face_detector is not None:
        face_results = face_detector.run(batched_tensor)
        for i in range(n):
            all_detections[i].extend(face_results[i])

    if lp_detector is not None:
        lp_results = lp_detector.run(batched_tensor)
        for i in range(n):
            all_detections[i].extend(lp_results[i])
    inference_ms = (time.time() - t_inf) * 1000

    t_blur = time.time()
    blurred = []
    for img, dets in zip(images_bgr, all_detections):
        blurred.append(visualize(img, dets, scale_factor))
    blur_ms = (time.time() - t_blur) * 1000

    return blurred, inference_ms, blur_ms


# ---------------------------------------------------------------------------
# Core workflow using AsyncImageFilter
# ---------------------------------------------------------------------------


def workflow(args: argparse.Namespace) -> None:
    debug_enc = getattr(args, "debug_encoding_only", False)
    if not debug_enc and args.face_model_path is None and args.lp_model_path is None:
        raise ValueError("Provide at least --face_model_path or --lp_model_path.")
    if not os.path.exists(args.input_vrs_path):
        raise FileNotFoundError(f"VRS file not found: {args.input_vrs_path}")

    os.makedirs(os.path.dirname(args.output_vrs_path) or ".", exist_ok=True)

    # --- build detectors ---
    face_detector: Optional[EgoblurDetector] = None
    lp_detector: Optional[EgoblurDetector] = None

    if not debug_enc:
        device = get_device()
        logger.warning(f"Using this device for inference: {device}")
        face_threshold = _get_threshold(
            args.camera_name, args.face_model_score_threshold, FACE_THRESHOLDS_GEN2
        )
        lp_threshold = _get_threshold(
            args.camera_name, args.lp_model_score_threshold, LP_THRESHOLDS_GEN2
        )

        use_gpu_resize = not args.aug_using_cpu

        if args.face_model_path is not None:
            face_detector = EgoblurDetector(
                model_path=args.face_model_path,
                device=device,
                detection_class=ClassID.FACE,
                score_threshold=face_threshold,
                nms_iou_threshold=args.nms_iou_threshold,
                resize_aug={
                    "min_size_test": RESIZE_MIN_GEN2,
                    "max_size_test": RESIZE_MAX_GEN2,
                },
                use_gpu_resize=use_gpu_resize,
            )

        if args.lp_model_path is not None:
            lp_detector = EgoblurDetector(
                model_path=args.lp_model_path,
                device=device,
                detection_class=ClassID.LICENSE_PLATE,
                score_threshold=lp_threshold,
                nms_iou_threshold=args.nms_iou_threshold,
                resize_aug={
                    "min_size_test": RESIZE_MIN_GEN2,
                    "max_size_test": RESIZE_MAX_GEN2,
                },
                use_gpu_resize=use_gpu_resize,
            )
    else:
        logger.info("debug_encoding_only: skipping model loading")

    # --- set up AsyncImageFilter ---
    # AsyncImageFilter handles:
    #   - Reading all streams from the input VRS
    #   - Iterating over image records for processing
    #   - Copying non-image streams (IMU, VIO, barometer, etc.) verbatim
    #   - Writing everything to the output VRS
    filtered_reader = FilteredFileReader(args.input_vrs_path)
    image_filter = AsyncImageFilter(filtered_reader)

    # Use a local temp file to avoid FUSE filesystem corruption.
    _tmp_fd, tmp_output_path = tempfile.mkstemp(
        suffix=".egoblur_tmp.vrs",
        prefix="egoblur_",
    )
    os.close(_tmp_fd)

    status = image_filter.create_output_file(tmp_output_path)
    if status != 0:
        raise RuntimeError(
            f"Failed to create output file: {tmp_output_path} (code={status})"
        )

    target_sids = set(args.stream_ids)
    batch_size = getattr(args, "batch_size", 1)
    logger.info(
        f"Processing VRS: {args.input_vrs_path}, "
        f"target camera streams: {sorted(target_sids)}, "
        f"batch_size: {batch_size}"
    )
    total_start = time.time()

    # Per-stream decoders, encoders, and timing stats.
    decoders: Dict[str, H265Decoder] = {}
    encoders: Dict[str, H265Encoder] = {}
    frame_counts: Dict[str, int] = {}
    all_timings: Dict[str, FrameTimings] = {}

    # Batch accumulators (across all streams when batch_size > 1).
    batch_buffers: List[ImageBuffer] = []
    batch_images: List[np.ndarray] = []
    batch_stream_ids: List[str] = []
    batch_decode_ms: List[float] = []

    def _flush_batch() -> None:
        """Run detection + blur on the accumulated batch, then encode and write.

        Images of different sizes (e.g. RGB 1920x2560 vs SLAM 512x512) cannot
        be stacked into a single tensor. We group by whether the stream is RGB
        or SLAM, run inference per group, then restore the original order for
        writing back via AsyncImageFilter.
        """
        if not batch_images:
            return

        # Group indices by stream type so same-size images are batched together.
        rgb_indices: List[int] = []
        slam_indices: List[int] = []
        for i, sid in enumerate(batch_stream_ids):
            if _is_rgb_stream(sid):
                rgb_indices.append(i)
            else:
                slam_indices.append(i)

        # Run detection + blur per group, then reassemble in original order.
        blurred: List[Optional[np.ndarray]] = [None] * len(batch_images)
        total_inference_ms = 0.0
        total_blur_ms = 0.0

        for group_indices in (rgb_indices, slam_indices):
            if not group_indices:
                continue
            group_images = [batch_images[i] for i in group_indices]
            if debug_enc:
                group_blurred = list(group_images)
                inf_ms, bl_ms = 0.0, 0.0
            else:
                group_blurred, inf_ms, bl_ms = _detect_and_blur_batch(
                    group_images,
                    face_detector,
                    lp_detector,
                    args.scale_factor_detections,
                )
            total_inference_ms += inf_ms
            total_blur_ms += bl_ms
            for gi, orig_idx in enumerate(group_indices):
                blurred[orig_idx] = group_blurred[gi]

        n_batch = len(batch_images)
        per_frame_inf = total_inference_ms / n_batch if n_batch else 0
        per_frame_blur = total_blur_ms / n_batch if n_batch else 0

        for idx, (img_buf, blurred_bgr, sid) in enumerate(
            zip(batch_buffers, blurred, batch_stream_ids)
        ):
            is_rgb = _is_rgb_stream(sid)
            h, w = blurred_bgr.shape[:2]

            # Lazily create encoder per stream (dimensions may differ).
            if sid not in encoders:
                encoders[sid] = H265Encoder(
                    width=w,
                    height=h,
                    is_rgb=is_rgb,
                    quality=args.encoding_quality,
                    preset=args.encoding_preset,
                )

            t_enc = time.time()
            if is_rgb:
                blurred_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)
                encode_input = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2YUV_I420)
            else:
                encode_input = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2GRAY)

            encoded_bytes = encoders[sid].encode_frame(encode_input)
            enc_ms = (time.time() - t_enc) * 1000

            # Record timings.
            timings = all_timings.setdefault(sid, FrameTimings())
            timings.decode_ms.append(batch_decode_ms[idx])
            timings.inference_ms.append(per_frame_inf)
            timings.blur_ms.append(per_frame_blur)
            timings.encode_ms.append(enc_ms)

            # Write processed image back via AsyncImageFilter.
            encoded_arr = np.frombuffer(encoded_bytes, dtype=np.uint8)
            out_buf = ImageBuffer(img_buf.spec, img_buf.record_index, encoded_arr)
            image_filter.write_processed_image(out_buf)

            frame_counts[sid] = frame_counts.get(sid, 0) + 1
            count = frame_counts[sid]
            if count % 10 == 0:
                elapsed = time.time() - total_start
                logger.info(f"[{sid}] frame {count}: ({count / elapsed:.1f} fps)")

        batch_buffers.clear()
        batch_images.clear()
        batch_stream_ids.clear()
        batch_decode_ms.clear()

    # --- process images ---
    with patch_instances(fields=PATCH_INSTANCES_FIELDS):
        for image_buffer in image_filter:
            record_info = image_filter.get_image_record_info(image_buffer.record_index)
            stream_id = record_info.streamId.get_numeric_name()

            # Only process target camera streams; pass others through unchanged.
            if stream_id not in target_sids:
                image_filter.write_processed_image(image_buffer)
                continue

            # Decode H.265 raw bytes using PyAV.
            is_rgb = _is_rgb_stream(stream_id)
            if stream_id not in decoders:
                decoders[stream_id] = H265Decoder(is_rgb=is_rgb)

            t_dec = time.time()
            raw_bytes = bytes(image_buffer.bytes)
            image_rgb = decoders[stream_id].decode_frame(raw_bytes)
            dec_ms = (time.time() - t_dec) * 1000

            if image_rgb is None:
                # Can't decode — pass through original bytes.
                image_filter.write_processed_image(image_buffer)
                continue

            if image_rgb.ndim == 2:
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            batch_buffers.append(image_buffer)
            batch_images.append(image_bgr)
            batch_stream_ids.append(stream_id)
            batch_decode_ms.append(dec_ms)

            if len(batch_images) >= batch_size:
                _flush_batch()

        # Process remaining partial batch.
        _flush_batch()

    # Flush remaining encoder packets.
    for sid, encoder in encoders.items():
        trailing = encoder.flush()
        if trailing:
            logger.info(f"Stream {sid}: flushed {len(trailing)} trailing encoder bytes")

    # --- finalize ---
    close_result = image_filter.close_file()
    if close_result != 0:
        raise RuntimeError(
            f"Failed to close VRS file: {tmp_output_path} (code={close_result})"
        )

    # Move temp file to final output path.
    try:
        shutil.move(tmp_output_path, args.output_vrs_path)
    except Exception:
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)
        raise

    total_frames = sum(frame_counts.values())
    total_time = time.time() - total_start
    for sid, count in frame_counts.items():
        logger.info(f"Stream {sid}: {count} frames")

    # Log per-stream timing breakdown.
    for sid in sorted(all_timings):
        summary = all_timings[sid].summary()
        if not summary:
            continue
        parts = []
        for phase in ("decode", "inference", "blur", "encode"):
            if phase in summary:
                s = summary[phase]
                parts.append(
                    f"{phase}: mean={s['mean']:.1f}ms p50={s['p50']:.1f}ms "
                    f"p95={s['p95']:.1f}ms total={s['total_s']:.1f}s"
                )
        logger.info(f"[{sid}] timing breakdown: {' | '.join(parts)}")

    logger.info(
        f"Done. {total_frames} frames across "
        f"{len(frame_counts)} streams in {total_time:.1f}s "
        f"({total_frames / max(total_time, 1e-9):.1f} fps) "
        f"-> {args.output_vrs_path}"
    )


def main() -> int:
    args = parse_args()
    workflow(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
