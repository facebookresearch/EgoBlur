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
and writes everything back into a single output VRS file.

All dependencies are OSS:
  - pyvrs (``pip install vrs``) for VRS reading and writing
  - egoblur (this package) for detection + blur
  - av (``pip install av``) for H.265 video encoding
"""

import argparse
import fractions
import os
import queue
import shutil
import tempfile
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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
from pyvrs import ImageConversion, SyncVRSReader
from pyvrs.writer import VRSStream, VRSWriter
from tqdm.auto import tqdm

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

# The OSS pyvrs writer supports named stream types registered in the StreamFactory.
# We first try the reader's original device name (which may match an internal
# registration), then fall back to the OSS Aria Gen2 camera type (correct
# RecordableTypeId + H.265).  If neither works, we raise an error.

# OSS pyvrs Aria Gen2 camera stream names (correct RecordableTypeId + H.265).
_ARIA_GEN2_STREAM_MAP: Dict[str, str] = {
    "214": "aria_gen2_rgb_camera",  # RecordableTypeId 214, H.265
    "1201": "aria_gen2_slam_camera",  # RecordableTypeId 1201, H.265
}

RGB_STREAM_PREFIX = "214"


def _is_rgb_stream(stream_id: str) -> bool:
    return stream_id.startswith(RGB_STREAM_PREFIX)


@dataclass(order=True)
class ProcessedFrame:
    """A processed frame ready for writing, sortable by (timestamp, sort_index)."""

    timestamp: float
    sort_index: int = field(compare=True)
    stream_id: str = field(compare=False)
    encoded_bytes: bytes = field(compare=False)
    metadata_items: List[Tuple[str, object]] = field(compare=False)

    @property
    def is_sentinel(self) -> bool:
        return self.timestamp == float("inf")


def _make_done_sentinel(stream_id: str) -> ProcessedFrame:
    """Create a sentinel ProcessedFrame that sorts after all real frames."""
    return ProcessedFrame(
        timestamp=float("inf"),
        sort_index=2**62,
        stream_id=stream_id,
        encoded_bytes=b"",
        metadata_items=[],
    )


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
# PyAV H.265 encoder (replaces Meta-internal xprs)
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


# Cache the check so we only probe once per process.
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
        # hevc_nvenc does not support "gray" pixel format — grayscale SLAM
        # frames must use CPU libx265.
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
                "preset": "p1",  # fastest NVENC preset
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
        """Encode a single numpy image and return the raw H.265 bytes.

        For RGB streams, *image* should already be in YUV420P planar layout
        (as produced by ``cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)``).
        For SLAM/grayscale streams, *image* should be a 2-D uint8 array.
        """
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
# VRS stream setup helpers
# ---------------------------------------------------------------------------


def _create_writer_stream(
    writer: VRSWriter,
    stream_id: str,
    flavor: str,
) -> "VRSStream":
    """Create a VRSStream using the Aria Gen2 camera stream type.

    Looks up the stream's RecordableTypeId prefix (e.g. ``"214"`` or ``"1201"``)
    in ``_ARIA_GEN2_STREAM_MAP`` and creates the corresponding writer stream.
    Raises ``ValueError`` if the prefix has no mapping.
    """
    type_prefix = stream_id.split("-")[0]
    aria_gen2_name = _ARIA_GEN2_STREAM_MAP.get(type_prefix)
    if aria_gen2_name is None:
        raise ValueError(
            f"Cannot create writer stream for {stream_id}: "
            f"no Aria Gen2 stream type registered for prefix '{type_prefix}'."
        )
    return writer.create_stream(aria_gen2_name, flavor=flavor)


def _setup_stream(
    reader: SyncVRSReader,
    writer: VRSWriter,
    stream_id: str,
) -> "VRSStream":
    """Create a VRSStream on *writer* that mirrors the source *stream_id*.

    Copies stream tags, state records, and configuration records.
    Uses the best available stream type for the correct RecordableTypeId
    and H.265 image content block declaration.
    """
    info = reader.get_stream_info(stream_id)
    original_flavor = info.get("flavor", "")
    original_device = info.get("device_name", "")

    stream = _create_writer_stream(writer, stream_id, original_flavor)

    # Preserve original stream identity as tags.
    stream.set_tag("original_stream_id", stream_id)
    if original_device:
        stream.set_tag("original_device_name", original_device)
    if original_flavor:
        stream.set_tag("original_flavor", original_flavor)

    # Copy stream-level tags.
    for key, val in reader.stream_tags[stream_id].items():
        stream.set_tag(key, val)

    # Copy state records.
    state_reader = reader.filtered_by_fields(
        stream_ids={stream_id}, record_types={"state"}
    )
    for record in state_reader:
        stream.create_state_record(record.timestamp)

    # Copy configuration records.
    cfg_reader = reader.filtered_by_fields(
        stream_ids={stream_id}, record_types={"configuration"}
    )
    for record in cfg_reader:
        config_md = stream.get_config_record_metadata()
        for metadata_block in record.metadata_blocks:
            for key, val in metadata_block.items():
                try:
                    config_md[key] = val
                except Exception:
                    logger.warning(
                        f"Skipping config key '{key}' for stream {stream_id}"
                    )
        stream.create_config_record(record.timestamp, config_md)

    return stream


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

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Batched detection + blur
# ---------------------------------------------------------------------------


def _detect_and_blur_batch(
    images_bgr: List[np.ndarray],
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    scale_factor: float,
) -> List[np.ndarray]:
    """Run batched face/LP detection and blur on a list of BGR images.

    Stacks per-image tensors into a single ``B x C x H x W`` batch and runs
    each detector once.  Returns a list of blurred BGR images (same length as
    *images_bgr*).
    """
    n = len(images_bgr)
    if n == 0:
        return []

    image_tensors = [get_image_tensor(img) for img in images_bgr]
    batched_tensor = torch.stack(image_tensors)

    all_detections: List[List[List[float]]] = [[] for _ in range(n)]

    if face_detector is not None:
        face_results = face_detector.run(batched_tensor)
        for i in range(n):
            all_detections[i].extend(face_results[i])

    if lp_detector is not None:
        lp_results = lp_detector.run(batched_tensor)
        for i in range(n):
            all_detections[i].extend(lp_results[i])

    return [
        visualize(img, dets, scale_factor)
        for img, dets in zip(images_bgr, all_detections)
    ]


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def _process_stream_worker(
    stream_id: str,
    input_vrs_path: str,
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    scale_factor: float,
    encoding_quality: int,
    encoding_preset: str,
    result_queue: "queue.Queue[ProcessedFrame]",
    frame_counter: List[int],
    counter_lock: threading.Lock,
    worker_timestamps: Dict[str, float],
    worker_ts_lock: threading.Lock,
    debug_encoding_only: bool = False,
    batch_size: int = 1,
) -> int:
    """Process all data records for a single stream.

    Each worker opens its own VRS reader, decoder, and encoder so that all
    heavy C-extension work (which releases the GIL) runs concurrently.
    Processed frames are pushed into *result_queue* for the main thread to
    write to the VRSWriter in timestamp order.

    When *batch_size* > 1, multiple decoded frames are accumulated and run
    through the detector in a single batched call before encoding each frame
    individually.

    Returns the number of frames processed.
    """
    reader = SyncVRSReader(input_vrs_path)
    reader.set_image_conversion(ImageConversion.RAW_BUFFER)

    is_rgb = _is_rgb_stream(stream_id)
    decoder = H265Decoder(is_rgb=is_rgb)
    encoder: Optional[H265Encoder] = None

    data_reader = reader.filtered_by_fields(
        stream_ids={stream_id}, record_types={"data"}
    )

    local_count = 0
    wall_start = time.time()

    # Batch accumulators: decoded images + per-frame metadata.
    batch_images: List[np.ndarray] = []
    batch_meta: List[Tuple[float, List[Tuple[str, object]]]] = []

    def _encode_and_enqueue(blurred_list: List[np.ndarray]) -> None:
        """Encode each blurred frame and push it into *result_queue*."""
        nonlocal encoder, local_count

        for (ts, md), blurred_bgr in zip(batch_meta, blurred_list):
            h, w = blurred_bgr.shape[:2]

            if encoder is None:
                encoder = H265Encoder(
                    width=w,
                    height=h,
                    is_rgb=is_rgb,
                    quality=encoding_quality,
                    preset=encoding_preset,
                )

            if is_rgb:
                blurred_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)
                encode_input = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2YUV_I420)
            else:
                encode_input = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2GRAY)

            encoded_bytes = encoder.encode_frame(encode_input)

            with counter_lock:
                global_index = frame_counter[0]
                frame_counter[0] += 1

            result_queue.put(
                ProcessedFrame(
                    timestamp=ts,
                    sort_index=global_index,
                    stream_id=stream_id,
                    encoded_bytes=encoded_bytes,
                    metadata_items=md,
                )
            )

            with worker_ts_lock:
                worker_timestamps[stream_id] = ts

            local_count += 1

            if local_count % 10 == 0:
                elapsed = time.time() - wall_start
                avg_ms = (elapsed / local_count) * 1000
                logger.info(
                    f"[{stream_id}] frame {local_count}: "
                    f"avg={avg_ms:.0f}ms "
                    f"({local_count / elapsed:.1f} fps)"
                )

    def _flush_batch() -> None:
        """Run detection + blur on the accumulated batch, then encode."""
        if not batch_images:
            return

        if debug_encoding_only:
            blurred = list(batch_images)
        else:
            blurred = _detect_and_blur_batch(
                batch_images, face_detector, lp_detector, scale_factor
            )

        _encode_and_enqueue(blurred)
        batch_images.clear()
        batch_meta.clear()

    for record in data_reader:
        if record.n_image_blocks == 0:
            continue

        # Decode H.265 raw bytes using PyAV.
        raw_bytes = bytes(record.image_blocks[0])
        image_rgb = decoder.decode_frame(raw_bytes)
        if image_rgb is None:
            continue

        if image_rgb.ndim == 2:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Collect metadata from the record.
        metadata_items: List[Tuple[str, object]] = []
        for metadata_block in record.metadata_blocks:
            for key, val in metadata_block.items():
                metadata_items.append((key, val))

        batch_images.append(image_bgr)
        batch_meta.append((record.timestamp, metadata_items))

        if len(batch_images) >= batch_size:
            _flush_batch()

    # Process remaining partial batch.
    _flush_batch()

    # Flush remaining encoder packets.
    if encoder is not None:
        trailing = encoder.flush()
        if trailing:
            logger.info(
                f"Stream {stream_id}: flushed {len(trailing)} trailing encoder bytes"
            )

    # Signal that this worker is done.
    result_queue.put(_make_done_sentinel(stream_id))
    return local_count


def _workflow_parallel(
    args: argparse.Namespace,
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    reader: SyncVRSReader,
    writer: VRSWriter,
    streams: Dict[str, "VRSStream"],
    target_sids: List[str],
) -> Dict[str, int]:
    """Multi-threaded processing: one worker per stream, main thread writes.

    NOTE: ``patch_instances`` modifies global TorchScript state and is NOT
    thread-safe, so we enter it once here in the main thread before any
    workers are launched.

    Flushing strategy: each worker updates its current timestamp in a shared
    dict after producing each frame.  The consumer periodically flushes up to
    ``min(active_worker_timestamps)`` which is guaranteed safe — no active
    worker can produce a record earlier than its current position.  After all
    workers finish, a final flush writes any remaining records.
    """
    total_data_frames = sum(
        int(reader.get_stream_info(sid).get("data_records_count", 0))
        for sid in target_sids
    )

    result_queue: queue.Queue[ProcessedFrame] = queue.Queue()
    frame_counter = [0]  # shared mutable counter for sort_index
    counter_lock = threading.Lock()

    # Per-worker timestamp tracking for safe incremental flushing.
    worker_timestamps: Dict[str, float] = {sid: 0.0 for sid in target_sids}
    worker_ts_lock = threading.Lock()
    active_workers: Set[str] = set(target_sids)

    num_workers = len(target_sids)
    logger.info(
        f"Parallel mode: launching {num_workers} worker threads "
        f"for streams {target_sids}"
    )

    frame_counts: Dict[str, int] = {sid: 0 for sid in target_sids}
    last_flushed_ts = 0.0

    with patch_instances(fields=PATCH_INSTANCES_FIELDS):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for sid in target_sids:
                future = executor.submit(
                    _process_stream_worker,
                    stream_id=sid,
                    input_vrs_path=args.input_vrs_path,
                    face_detector=face_detector,
                    lp_detector=lp_detector,
                    scale_factor=args.scale_factor_detections,
                    encoding_quality=args.encoding_quality,
                    encoding_preset=args.encoding_preset,
                    result_queue=result_queue,
                    frame_counter=frame_counter,
                    counter_lock=counter_lock,
                    worker_timestamps=worker_timestamps,
                    worker_ts_lock=worker_ts_lock,
                    debug_encoding_only=getattr(args, "debug_encoding_only", False),
                    batch_size=getattr(args, "batch_size", 1),
                )
                futures[future] = sid

            # Main thread: consume processed frames and flush incrementally.
            workers_done = 0
            frames_written = 0

            while workers_done < num_workers:
                item = result_queue.get()
                if item.is_sentinel:
                    workers_done += 1
                    active_workers.discard(item.stream_id)
                    continue

                pf: ProcessedFrame = item
                stream = streams[pf.stream_id]
                data_md = stream.get_data_record_metadata()
                for key, val in pf.metadata_items:
                    try:
                        data_md[key] = val
                    except Exception:
                        pass

                encoded_arr = np.frombuffer(pf.encoded_bytes, dtype=np.uint8)
                stream.create_data_record(pf.timestamp, data_md, encoded_arr)

                frame_counts[pf.stream_id] += 1
                frames_written += 1

                # Periodically flush records up to the safe frontier.
                if frames_written % 50 == 0:
                    with worker_ts_lock:
                        active_ts = [worker_timestamps[sid] for sid in active_workers]
                    # Only flush if all active workers have started producing
                    # (timestamp > 0). A worker at 0 hasn't emitted its first
                    # frame yet, so we can't safely advance the frontier.
                    if active_ts and all(t > 0 for t in active_ts):
                        frontier_ts = min(active_ts)
                        if frontier_ts > last_flushed_ts:
                            writer.flush_records(frontier_ts)
                            last_flushed_ts = frontier_ts

                if frames_written % 100 == 0:
                    logger.info(
                        f"Consumer: {frames_written}/{total_data_frames} records created"
                    )

            # Check for any worker exceptions.
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    count = future.result()
                    logger.info(f"Worker {sid}: processed {count} frames")
                except Exception:
                    logger.exception(f"Worker {sid} failed")
                    raise

    # Final flush: all workers are done, write any remaining buffered records.
    max_ts = max(worker_timestamps.values()) if worker_timestamps else 0.0
    if max_ts > last_flushed_ts:
        logger.info(f"Final flush: writing remaining records up to ts={max_ts:.6f}")
        writer.flush_records(max_ts)

    return frame_counts


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
            )
    else:
        logger.info("debug_encoding_only: skipping model loading")

    # --- open source VRS and set up output writer ---
    reader = SyncVRSReader(args.input_vrs_path)
    reader.set_image_conversion(ImageConversion.RAW_BUFFER)

    available_sids: Set[str] = set(reader.stream_ids)
    target_sids = [s for s in args.stream_ids if s in available_sids]
    if not target_sids:
        raise ValueError(
            f"None of the requested stream IDs {args.stream_ids} exist in "
            f"{args.input_vrs_path}. Available: {sorted(available_sids)}"
        )

    # Check whether there are non-camera streams to passthrough.
    non_camera_sids = available_sids - set(target_sids)
    has_passthrough = bool(non_camera_sids)
    if not has_passthrough:
        logger.info("All streams are camera streams — no passthrough needed")
    else:
        logger.info(
            f"Passthrough: {len(non_camera_sids)} non-camera streams "
            f"will be copied verbatim"
        )

    # Always use a local temp directory for the writer output because
    # pyvrs VRSWriter uses seek-based I/O that can corrupt on FUSE
    # filesystems (e.g. Manifold).
    _tmp_fd, tmp_output_path = tempfile.mkstemp(
        suffix=".egoblur_tmp.vrs",
        prefix="egoblur_",
    )
    os.close(_tmp_fd)

    writer = VRSWriter(tmp_output_path)

    # Copy file-level tags from source (device_type, calib_json, etc.).
    for key, value in reader.file_tags.items():
        writer.set_tag(key, value)

    # Create all streams on the single writer.
    streams: Dict[str, "VRSStream"] = {}
    for sid in target_sids:
        streams[sid] = _setup_stream(reader, writer, sid)

    # Register passthrough streams for verbatim copy (before first flush).
    if has_passthrough:
        writer.add_verbatim_copy_streams(reader._reader, sorted(non_camera_sids))

    batch_size = getattr(args, "batch_size", 1)
    logger.info(
        f"Processing VRS: {args.input_vrs_path}, streams: {target_sids}, "
        f"batch_size: {batch_size}"
    )
    total_start = time.time()

    # --- process camera streams ---
    frame_counts = _workflow_parallel(
        args, face_detector, lp_detector, reader, writer, streams, target_sids
    )

    # --- finalize ---
    # Copy passthrough stream records into the writer before closing.
    if has_passthrough:
        logger.info("Copying passthrough streams verbatim...")
        writer.copy_verbatim_records()

    close_result = writer.close()
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
    passthrough_str = " (with passthrough)" if has_passthrough else ""
    logger.info(
        f"Done{passthrough_str}. {total_frames} frames across "
        f"{len(target_sids)} streams in {total_time:.1f}s "
        f"({total_frames / max(total_time, 1e-9):.1f} fps) "
        f"-> {args.output_vrs_path}"
    )


def main() -> int:
    args = parse_args()
    workflow(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
