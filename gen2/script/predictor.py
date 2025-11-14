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

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from detectron2.structures import Boxes, Instances
from detectron2.utils import (
    convert_scripted_instances,
    detector_postprocess,
    ResizeShortestEdge,
)
from torch.jit import RecursiveScriptModule


class ClassID(int, Enum):
    LICENSE_PLATE = 1
    FACE = 0


LABEL_TO_ID: Dict[str, int] = {
    "face": ClassID.FACE.value,
    "license_plate": ClassID.LICENSE_PLATE.value,
}

PATCH_INSTANCES_FIELDS = {
    "proposal_boxes": Boxes,
    "objectness_logits": torch.Tensor,
    "pred_boxes": Boxes,
    "scores": torch.Tensor,
    "pred_classes": torch.Tensor,
    "pred_masks": torch.Tensor,
}

MIN_MODEL_SCORE_THRESHOLD = 0.0
MAX_MODEL_SCORE_THRESHOLD = 1.0


@dataclass
class FrameDetections:
    timestamp_s: float
    rotation_angle: float
    stream_id: str
    face_bboxes: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 4)))
    face_scores: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 1)))
    face_classes: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 1)))
    lp_bboxes: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 4)))
    lp_scores: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 1)))
    lp_classes: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 1)))


class EgoblurDetector:
    def __init__(
        self,
        model_path: str,
        device: str,
        detection_class: ClassID,
        score_threshold: float,
        nms_iou_threshold: float,
        tscript_type: str = "script",
        image_format: str = "BGR",
        resize_aug: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model_path: Absolute path to a TorchScript file (``.pt``) on disk.
            device: Torch device string (e.g. ``cpu`` or ``cuda:0``) where the model runs.
            detection_class: Target ``ClassID`` this detector should filter for.
            score_threshold: Score threshold in ``[0.0, 1.0]`` applied after NMS.
            nms_iou_threshold: IoU threshold in ``[0.0, 1.0]`` used for torchvision NMS.
            tscript_type: ``"script"`` (batched) or ``"trace"`` (single-image only).
            image_format: Incoming image layout string, ``"BGR"`` or ``"RGB"``.
            resize_aug: Optional dictionary with keys ``min_size_test`` and ``max_size_test`` to
                configure a ``ResizeShortestEdge`` augmentation. If omitted, no resize augmentation
                is applied.
        """
        self.detection_class = detection_class
        self.device_str = self._validate_model_device(device)
        self.device = torch.device(self.device_str)
        self._model_torchscript_file = self._validate_model_torchscript_file(model_path)
        self._model_score_threshold = self._validate_model_score_threshold(
            score_threshold
        )
        self._nms_iou_threshold = self._validate_nms_iou_threshold(nms_iou_threshold)
        assert tscript_type in {
            "trace",
            "script",
        }, "tscript_type must be trace or script"
        self.tscript_type = tscript_type
        assert image_format in {"BGR", "RGB"}
        self.image_format = image_format

        category_str = "face" if detection_class == ClassID.FACE else "license_plate"
        self._class = LABEL_TO_ID[category_str]

        self._model = self._load_torchscript_model()

        if resize_aug is not None:
            assert "min_size_test" in resize_aug, "min_size_test must be in resize_aug"
            assert "max_size_test" in resize_aug, "max_size_test must be in resize_aug"
            self.aug = ResizeShortestEdge(
                short_edge_length=[
                    resize_aug["min_size_test"],
                    resize_aug["min_size_test"],
                ],
                max_size=resize_aug["max_size_test"],
            )
        else:
            self.aug = None

    @staticmethod
    def _validate_model_torchscript_file(model_torchscript_file: str) -> str:
        """
        Args:
            model_torchscript_file: Candidate filesystem path to the TorchScript file.

        Returns:
            Same path once validated to be non-empty.

        Raises:
            ValueError: If the provided path is empty.
        """
        if not model_torchscript_file:
            raise ValueError(
                "required parameter model_torchscript_file for EgoblurDetector is empty or None, "
                f"provided value {model_torchscript_file}"
            )
        return model_torchscript_file

    @staticmethod
    def _validate_model_device(model_device: str) -> str:
        """
        Args:
            model_device: Torch device string (e.g. ``"cpu"`` or ``"cuda:0"``).

        Returns:
            Same device string after validation.

        Raises:
            ValueError: If the device string is empty.
        """
        if not model_device:
            raise ValueError(
                "required parameter model_device for EgoblurDetector is empty or None, "
                f"provided value {model_device}"
            )
        return model_device

    @staticmethod
    def _validate_model_score_threshold(model_score_threshold: float) -> float:
        """
        Args:
            model_score_threshold: Desired score threshold.

        Returns:
            Threshold if it lies within ``[MIN_MODEL_SCORE_THRESHOLD, MAX_MODEL_SCORE_THRESHOLD]``.

        Raises:
            ValueError: When the value is ``None`` or out of range.
        """
        if not (
            MIN_MODEL_SCORE_THRESHOLD
            <= model_score_threshold
            <= MAX_MODEL_SCORE_THRESHOLD
        ):
            raise ValueError(
                "required parameter score_threshold for EgoblurDetector is outside a valid range "
                f"of {MIN_MODEL_SCORE_THRESHOLD} to {MAX_MODEL_SCORE_THRESHOLD}, provided value {model_score_threshold}"
            )
        return model_score_threshold

    @staticmethod
    def _validate_nms_iou_threshold(nms_iou_threshold: float) -> float:
        """
        Args:
            nms_iou_threshold: IoU threshold used for NMS.

        Returns:
            Threshold if it lies within ``[0.0, 1.0]``.

        Raises:
            ValueError: When the value is ``None`` or outside the valid range.
        """
        if not (0.0 <= nms_iou_threshold <= 1.0):
            raise ValueError(
                "required parameter nms_iou_threshold for EgoblurDetector must be between 0.0 and 1.0, "
                f"provided value {nms_iou_threshold}"
            )
        return nms_iou_threshold

    def _load_torchscript_model(self) -> RecursiveScriptModule:
        """
        Loads the serialized TorchScript model.

        Returns:
            ``RecursiveScriptModule`` moved to ``self.device``.

        Raises:
            RuntimeError: If loading or device placement fails.
        """
        try:
            model = torch.jit.load(self._model_torchscript_file, map_location="cpu")
            model.to(self.device)
        except Exception as exc:
            message = (
                "Failed to instantiate torchscript model from "
                f"{self._model_torchscript_file}: {exc}"
            )
            raise RuntimeError(message) from exc
        return model

    def transform_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Args:
            img: Image as an ``H x W x C`` NumPy array in RGB or grayscale.

        Returns:
            Torch tensor with shape ``C x H x W`` located on ``self.device``.
        """
        if self.image_format == "BGR":
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.image_format == "RGB":
            if len(img.shape) <= 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported image format: {self.image_format}")

        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img)
        return img_tensor.to(self.device)

    def pre_process(
        self, bgr_image_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Args:
            bgr_image_batch: Batched tensor shaped ``B x C x H x W`` in BGR order (uint8).

        Returns:
            Tuple of:
                * ``torch.Tensor`` – stacked batch ready for the model, dtype inherited from input.
                * ``List[Tuple[int, int]]`` – original ``(H, W)`` per image.
                * ``List[Tuple[int, int]]`` – model-input ``(H, W)`` after augmentation.
        """
        img_batch: List[np.ndarray] = [
            im.transpose(1, 2, 0)[:, :, ::-1] for im in bgr_image_batch.cpu().numpy()
        ]
        orig_img_hw_list = [img.shape[:2] for img in img_batch]

        if self.aug is not None:
            img_batch = [
                self.aug.get_transform(img).apply_image(img) for img in img_batch
            ]
        model_input_hw_list = [img.shape[:2] for img in img_batch]

        img_tensor_list = [self.transform_image(im_np) for im_np in img_batch]
        return torch.stack(img_tensor_list), orig_img_hw_list, model_input_hw_list

    def inference(
        self, image_batch: torch.Tensor
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            image_batch: Batched tensor shaped ``B x C x H x W`` already on ``self.device``.

        Returns:
            For ``tscript_type == "script"``: list of scripted ``Instances`` (length ``B``).
            For ``tscript_type == "trace"``: list of tensors representing model heads.

        Raises:
            ValueError: If batch size > 1 for trace models or ``tscript_type`` is unknown.
        """
        batchsize = len(image_batch)
        if self.tscript_type == "trace":
            if batchsize != 1:
                raise ValueError(
                    f"trace model only supports batchsize 1, got {batchsize}"
                )
            with torch.no_grad():
                preds = self._model(image_batch[0])
            return [pred.unsqueeze(0) for pred in preds]

        if self.tscript_type != "script":
            raise ValueError(f"Unknown tscript_type {self.tscript_type}")

        with torch.no_grad():
            batch_list = [{"image": im_t} for im_t in image_batch]
            preds = self._model.inference(batch_list, do_postprocess=False)
            if len(preds) != batchsize:
                raise ValueError(
                    f"expected {batchsize} outputs, got {len(preds)} from model"
                )
        return preds

    def get_detections(
        self,
        output_tensor: Union[List[torch.Tensor], torch.Tensor],
        timestamp_s: float,
        stream_id: str,
        rotation_angle: float,
        model_input_hw_list: List[Tuple[int, int]],
        target_img_hw_list: List[Tuple[int, int]],
    ) -> List[Optional[FrameDetections]]:
        """
        Args:
            output_tensor: Raw model outputs from ``inference``; format depends on model type.
            timestamp_s: Frame timestamp in seconds.
            stream_id: Identifier for the stream/frame source.
            rotation_angle: Rotation angle applied to the source frame.
            model_input_hw_list: Model input ``(H, W)`` per batch item.
            target_img_hw_list: Target/original ``(H, W)`` per batch item.

        Returns:
            List with up to ``B`` ``FrameDetections`` objects (missing entries are skipped when empty).

        Raises:
            ValueError: On batch size mismatches.
        """
        batch_boxes, batch_scores = self._post_process(
            output_tensor,
            model_input_hw_list=model_input_hw_list,
            target_img_hw_list=target_img_hw_list,
        )

        batchsize = len(model_input_hw_list)
        if not (len(batch_boxes) == len(batch_scores) == batchsize):
            raise ValueError(
                "Mismatch between batch sizes of boxes, scores, and inputs: "
                f"{len(batch_boxes)=}, {len(batch_scores)=}, expected {batchsize}"
            )
        if len(target_img_hw_list) != batchsize:
            raise ValueError(
                "Mismatch between batch sizes of targets and inputs: "
                f"{len(target_img_hw_list)=}, expected {batchsize}"
            )

        detections_batch: List[Optional[FrameDetections]] = []
        for boxes, scores in zip(batch_boxes, batch_scores):
            if not boxes.any():
                continue

            detections = FrameDetections(
                timestamp_s=timestamp_s,
                rotation_angle=rotation_angle,
                stream_id=stream_id,
            )
            if self._class == ClassID.FACE:
                detections.face_bboxes = boxes
                detections.face_scores = scores
                detections.face_classes = np.ones_like(scores).astype(int) * self._class
            elif self._class == ClassID.LICENSE_PLATE:
                detections.lp_bboxes = boxes
                detections.lp_scores = scores
                detections.lp_classes = np.ones_like(scores).astype(int) * self._class
            else:
                raise ValueError(f"Unknown class {self._class}")
            detections_batch.append(detections)
        return detections_batch

    def _post_process(
        self,
        preds0: Union[List[torch.Tensor], torch.Tensor],
        model_input_hw_list: List[Tuple[int, int]],
        target_img_hw_list: List[Tuple[int, int]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Args:
            preds0: Outputs from ``inference`` (scripted Instances or raw tensors).
            model_input_hw_list: Model input ``(H, W)`` per batch item.
            target_img_hw_list: Target/original ``(H, W)`` per batch item.

        Returns:
            Tuple of two lists (length ``B``):
                * NumPy arrays of post-processed boxes in XYXY order.
                * NumPy arrays of per-box scores.
        """
        preds0 = deepcopy(preds0)
        preds: List[Instances] = []
        if self.tscript_type == "trace":
            boxes = scores = labels = None
            if len(preds0) == 4:
                boxes, _, scores, _ = preds0
            elif len(preds0) == 5:
                boxes, labels, _, scores, _ = preds0
            else:
                raise ValueError(
                    f"unexpected number of outputs from model {len(preds0)}, expected 4 or 5."
                )
            batchsize = len(model_input_hw_list)
            for bi in range(batchsize):
                input_h, input_w = model_input_hw_list[bi]
                pred = Instances((input_h, input_w))
                pred.pred_boxes = Boxes(boxes[bi])
                pred.scores = scores[bi]
                if labels is not None:
                    pred.pred_classes = labels[bi]
                preds.append(pred)
        else:
            if self.tscript_type != "script":
                raise ValueError(f"Unknown tscript_type {self.tscript_type}")
            preds = [convert_scripted_instances(pred) for pred in preds0]

        if self.aug is not None:
            processed_preds: List[Instances] = []
            for i, pred in enumerate(preds):
                orig_h, orig_w = target_img_hw_list[i]
                processed_preds.append(detector_postprocess(pred, orig_h, orig_w))
            preds = processed_preds

        pred_boxes_batch: List[torch.Tensor] = []
        pred_scores_batch: List[torch.Tensor] = []
        for pred in preds:
            boxes = pred.pred_boxes.tensor
            scores = pred.scores
            if pred.has("pred_classes"):
                labels = pred.pred_classes
                boxes = boxes[labels == self._class]
                scores = scores[labels == self._class]
            pred_boxes_batch.append(boxes)
            pred_scores_batch.append(scores)

        output_boxes_batch: List[np.ndarray] = []
        output_scores_batch: List[np.ndarray] = []
        for boxes, scores in zip(pred_boxes_batch, pred_scores_batch):
            if boxes.numel() == 0:
                output_boxes_batch.append(boxes.cpu().detach().numpy())
                output_scores_batch.append(scores.cpu().detach().numpy())
                continue

            nms_keep_idx = torchvision.ops.nms(boxes, scores, self._nms_iou_threshold)
            boxes = boxes[nms_keep_idx]
            scores = scores[nms_keep_idx]

            boxes = boxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()

            score_keep_idx = np.where(scores > self._model_score_threshold)[0]
            boxes = boxes[score_keep_idx]
            scores = scores[score_keep_idx]

            output_boxes_batch.append(boxes)
            output_scores_batch.append(scores)

        return output_boxes_batch, output_scores_batch

    def run(self, image_tensor: torch.Tensor) -> List[List[List[float]]]:
        """
        Args:
            image_tensor: Either single image ``C x H x W`` or batch ``B x C x H x W`` tensor.
                Batched inputs are fully supported; callers that only handle single images can
                still pass rank-3 tensors and allow this method to add the batch dimension.

        Returns:
            List of length ``B`` containing detection boxes as ``List[List[float]]`` in XYXY order.

        Raises:
            ValueError: If the input tensor rank is not 3 or 4.
        """
        if image_tensor.ndim == 3:
            batched = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 4:
            batched = image_tensor
        else:
            raise ValueError(
                "Expected image tensor with shape CxHxW or BxCxHxW, "
                f"got {image_tensor.shape}"
            )

        img_batch, orig_img_hw_list, model_input_hw_list = self.pre_process(batched)
        preds = self.inference(img_batch)
        detections_batch = self.get_detections(
            output_tensor=preds,
            timestamp_s=0.0,
            stream_id="",
            rotation_angle=0.0,
            model_input_hw_list=model_input_hw_list,
            target_img_hw_list=orig_img_hw_list,
        )

        batch_results: List[List[List[float]]] = []
        for detections in detections_batch:
            if detections is None:
                batch_results.append([])
                continue

            boxes = (
                detections.face_bboxes
                if self.detection_class == ClassID.FACE
                else detections.lp_bboxes
            )

            if boxes.size == 0:
                batch_results.append([])
                continue

            batch_results.append(boxes.tolist())

        return batch_results
