from functools import lru_cache
from typing import List

import cv2
import numpy as np
import torch
import torchvision


# ------------------------------------------------------------------------------
class ImageAnonymizer:

    def __init__(
        self,
        face_model_path: str,
        lp_model_path: str,
        face_model_score_threshold: float = 0.9,
        lp_model_score_threshold: float = 0.9,
        nms_iou_threshold: float = 0.3,
        scale_factor_detections: float = 1.0,
    ):
        """
        Initialize the ImageAnonymizer class.

        parameter face_model_path: path to face detector model to perform face detections
        parameter lp_model_path: path to face detector model to perform face detections
        parameter face_model_score_threshold: face model score threshold to filter out low confidence detection
        parameter lp_model_score_threshold: license plate model score threshold to filter out low confidence detection
        parameter nms_iou_threshold: NMS iou threshold
        parameter scale_factor_detections: scale detections by the given factor to allow blurring more area
        """
        if face_model_path is not None:
            face_detector = torch.jit.load(
                face_model_path, map_location="cpu"
            ).to(self.get_device())
            face_detector.eval()
        else:
            face_detector = None

        if lp_model_path is not None:
            lp_detector = torch.jit.load(lp_model_path, map_location="cpu").to(
                self.get_device()
            )
            lp_detector.eval()
        else:
            lp_detector = None

        self._face_detector = face_detector
        self._lp_detector = lp_detector
        self._face_model_score_threshold = face_model_score_threshold
        self._lp_model_score_threshold = lp_model_score_threshold
        self._nms_iou_threshold = nms_iou_threshold
        self._scale_factor_detections = scale_factor_detections

    # ----------------
    @lru_cache
    def get_device(self) -> str:
        """
        Return the device type
        """
        return (
            "cpu"
            if not torch.cuda.is_available()
            else f"cuda:{torch.cuda.current_device()}"
        )

    # ----------------
    def get_image_tensor(self, bgr_image: np.ndarray) -> torch.Tensor:
        """
        parameter bgr_image: image on which we want to make detections

        Return the image tensor
        """
        bgr_image_transposed = np.transpose(bgr_image, (2, 0, 1))
        image_tensor = torch.from_numpy(bgr_image_transposed).to(
            self.get_device()
        )

        return image_tensor

    def get_detections(
        self,
        detector: torch.jit._script.RecursiveScriptModule,
        image_tensor: torch.Tensor,
        model_score_threshold: float,
        nms_iou_threshold: float,
    ) -> List[List[float]]:
        """
        parameter detector: Torchscript module to perform detections
        parameter image_tensor: image tensor on which we want to make detections
        parameter model_score_threshold: model score threshold to filter out low confidence detection
        parameter nms_iou_threshold: NMS iou threshold to filter out low confidence overlapping boxes

        Returns the list of detections
        """
        with torch.no_grad():
            detections = detector(image_tensor)

        boxes, _, scores, _ = detections  # returns boxes, labels, scores, dims

        nms_keep_idx = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
        boxes = boxes[nms_keep_idx]
        scores = scores[nms_keep_idx]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        score_keep_idx = np.where(scores > model_score_threshold)[0]
        boxes = boxes[score_keep_idx]
        return boxes.tolist()

    def scale_box(
        self,
        box: List[List[float]],
        max_width: int,
        max_height: int,
        scale: float,
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

    def visualize(
        self,
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
                box = self.scale_box(
                    box, image.shape[1], image.shape[0], scale_factor_detections
                )
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            w = x2 - x1
            h = y2 - y1

            ksize = (image.shape[0] // 2, image.shape[1] // 2)
            image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)
            cv2.ellipse(
                mask, (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 0), 255, -1
            )

        inverse_mask = cv2.bitwise_not(mask)
        image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
        image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
        image = cv2.add(image_bg, image_fg)

        return image

    def visualize_image(
        self,
        bgr_image: np.ndarray,
    ) -> np.ndarray:
        """
        parameter bgr_image: bgr formatted image to be anonymized

        Perform detections on the input image and save the output image at the given path.
        """
        image = bgr_image.copy()
        image_tensor = self.get_image_tensor(bgr_image)
        image_tensor_copy = image_tensor.clone()
        detections = []

        # get face detections
        if self._face_detector is not None:
            detections.extend(
                self.get_detections(
                    self._face_detector,
                    image_tensor,
                    self._face_model_score_threshold,
                    self._nms_iou_threshold,
                )
            )

        # get license plate detections
        if self._lp_detector is not None:
            detections.extend(
                self.get_detections(
                    self._lp_detector,
                    image_tensor_copy,
                    self._lp_model_score_threshold,
                    self._nms_iou_threshold,
                )
            )

        image = self.visualize(
            image,
            detections,
            self._scale_factor_detections,
        )

        return image
