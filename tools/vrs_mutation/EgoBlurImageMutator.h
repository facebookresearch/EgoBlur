/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/types.h>
#include <vrs/RecordFormat.h> // @manual

#include <projectaria_tools/tools/samples/vrs_mutation/ImageMutationFilterCopier.h> // @manual

#include <torch/script.h>
#include <torch/serialize.h> // @manual
#include <torch/torch.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include <c10/cuda/CUDACachingAllocator.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace EgoBlur {

struct EgoBlurImageMutator : public vrs::utils::UserDefinedImageMutator {
  // Inherit from UserDefinedImageMutator as defined in projectaria_tools to
  // blur detected faces/license plates. This class implements the logic to run
  // model inference and performs blurring on VRS frame by frame and saves the
  // output as a VRS file.
  std::shared_ptr<torch::jit::script::Module> faceModel_;
  std::shared_ptr<torch::jit::script::Module> licensePlateModel_;
  float faceModelConfidenceThreshold_;
  float licensePlateModelConfidenceThreshold_;
  float scaleFactorDetections_;
  float nmsThreshold_;
  bool useGPU_;
  bool clockwise90Rotation_;
  std::unordered_map<std::string, std::unordered_map<std::string, int>> stats_;
  torch::Device device_ = torch::kCPU;

  explicit EgoBlurImageMutator(
      const std::string& faceModelPath = "",
      const float faceModelConfidenceThreshold = 0.1,
      const std::string& licensePlateModelPath = "",
      const float licensePlateModelConfidenceThreshold = 0.1,
      const float scaleFactorDetections = 1.15,
      const float nmsThreshold = 0.3,
      const bool useGPU = true,
      const bool clockwise90Rotation = true)
      : faceModelConfidenceThreshold_(faceModelConfidenceThreshold),
        licensePlateModelConfidenceThreshold_(
            licensePlateModelConfidenceThreshold),
        scaleFactorDetections_(scaleFactorDetections),
        nmsThreshold_(nmsThreshold),
        useGPU_(useGPU),
        clockwise90Rotation_(clockwise90Rotation) {
    device_ = getDevice();
    std::cout << "attempting to load ego blur face model: " << faceModelPath
              << std::endl;

    if (!faceModelPath.empty()) {
      faceModel_ = loadModel(faceModelPath);
    }

    std::cout << "attempting to load ego blur license plate model: "
              << licensePlateModelPath << std::endl;

    if (!licensePlateModelPath.empty()) {
      licensePlateModel_ = loadModel(licensePlateModelPath);
    }
  }

  std::shared_ptr<torch::jit::script::Module> loadModel(
      const std::string& path) {
    std::shared_ptr<torch::jit::script::Module> model;
    try {
      model = std::make_shared<torch::jit::script::Module>();
      // patternlint-disable-next-line no-torch-low-level-api
      *model = torch::jit::load(path);
      std::cout << "Loaded model: " << path << std::endl;
      model->to(device_);
      model->eval();
    } catch (const c10::Error&) {
      std::cout << "Failed to load model: " << path << std::endl;
      throw;
    }
    return model;
  }

  at::DeviceType getDevice() const {
    if (useGPU_ && torch::cuda::is_available()) {
      // using GPU
      return torch::kCUDA;
    } else {
      // using CPU
      return torch::kCPU;
    }
  }

  torch::Tensor filterDetections(
      c10::intrusive_ptr<c10::ivalue::Tuple> detections,
      float scoreThreshold) const {
    // filter prediction based of confidence scores, we have scores at index 2
    torch::Tensor scoreThresholdMask =
        torch::gt(
            detections->elements().at(2).toTensor(),
            torch::tensor(scoreThreshold))
            .detach();
    // we have boxes at index 0
    torch::Tensor filteredBoundingBoxes = detections->elements()
                                              .at(0)
                                              .toTensor()
                                              .index({scoreThresholdMask})
                                              .detach();
    torch::Tensor filteredBoundingBoxesScores = detections->elements()
                                                    .at(2)
                                                    .toTensor()
                                                    .index({scoreThresholdMask})
                                                    .detach();

    // filter out overlapping detections by performing NMS
    torch::Tensor filteredBoundingBoxesPostNMS =
        performNMS(
            filteredBoundingBoxes, filteredBoundingBoxesScores, nmsThreshold_)
            .detach();
    scoreThresholdMask.reset();
    filteredBoundingBoxes.reset();
    filteredBoundingBoxesScores.reset();
    return filteredBoundingBoxesPostNMS;
  }

  // Define a custom NMS function
  torch::Tensor performNMS(
      const torch::Tensor& boxes,
      const torch::Tensor& scores,
      float overlapThreshold) const {
    // Convert tensors to CPU
    torch::Tensor boxesCPU = boxes.to(torch::kCPU).detach();
    torch::Tensor scoresCPU = scores.to(torch::kCPU).detach();

    // Get the number of bounding boxes
    int numBoxes = boxesCPU.size(0);

    // Extract bounding box coordinates
    auto boxesAccessor = boxesCPU.accessor<float, 2>();
    auto scoresAccessor = scoresCPU.accessor<float, 1>();

    std::vector<bool> picked(numBoxes, false);

    for (int i = 0; i < numBoxes; ++i) {
      if (!picked[i]) {
        for (int j = i + 1; j < numBoxes; ++j) {
          if (!picked[j]) {
            float x1 = std::max(boxesAccessor[i][0], boxesAccessor[j][0]);
            float y1 = std::max(boxesAccessor[i][1], boxesAccessor[j][1]);
            float x2 = std::min(boxesAccessor[i][2], boxesAccessor[j][2]);
            float y2 = std::min(boxesAccessor[i][3], boxesAccessor[j][3]);

            float intersection =
                std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float iou = intersection /
                ((boxesAccessor[i][2] - boxesAccessor[i][0]) *
                     (boxesAccessor[i][3] - boxesAccessor[i][1]) +
                 (boxesAccessor[j][2] - boxesAccessor[j][0]) *
                     (boxesAccessor[j][3] - boxesAccessor[j][1]) -
                 intersection);

            if (iou > overlapThreshold) {
              if (scoresAccessor[i] > scoresAccessor[j]) {
                picked[j] = true;
              } else {
                picked[i] = true;
              }
            }
          }
        }
      }
    }

    std::vector<int> selectedIndices;
    for (int i = 0; i < numBoxes; ++i) {
      if (!picked[i]) {
        selectedIndices.push_back(i);
      }
    }

    torch::Tensor filteredBoundingBoxes =
        torch::index_select(
            boxes.to(torch::kCPU),
            0,
            torch::from_blob(
                selectedIndices.data(),
                {static_cast<long>(selectedIndices.size())},
                torch::kInt))
            .detach();

    boxesCPU.reset();
    scoresCPU.reset();
    return filteredBoundingBoxes;
  }

  static std::vector<float> scaleBox(
      const std::vector<float>& box,
      int maxWidth,
      int maxHeight,
      float scale) {
    // Extract x1, y1, x2, and y2 from the input box.
    float x1 = box[0];
    float y1 = box[1];
    float x2 = box[2];
    float y2 = box[3];
    float w = x2 - x1;
    float h = y2 - y1;

    // Calculate the center point of the box.
    float xc = x1 + (w / 2);
    float yc = y1 + (h / 2);
    // Scale the width and height of the box.
    w = scale * w;
    h = scale * h;
    // Update the coordinates of the box to fit within the maximum dimensions.
    x1 = std::max(xc - (w / 2), 0.0f);
    y1 = std::max(yc - (h / 2), 0.0f);
    x2 = std::min(xc + (w / 2), static_cast<float>(maxWidth));
    y2 = std::min(yc + (h / 2), static_cast<float>(maxHeight));
    // Return the scaled box as a vector of vectors.
    return {x1, y1, x2, y2};
  }

  cv::Mat blurImage(
      const cv::Mat& image,
      const std::vector<torch::Tensor>& detections,
      float scale) {
    // Use the mask to combine the original and blurred images
    cv::Mat response = image.clone();
    cv::Mat mask;
    if (image.channels() == 3) {
      mask = cv::Mat::zeros(image.size(), CV_8UC3);
    } else {
      mask = cv::Mat::zeros(image.size(), CV_8UC1);
    }
    for (const auto& detection : detections) {
      for (auto& box : detection.unbind()) {
        std::vector<float> boxVector(
            box.data_ptr<float>(), box.data_ptr<float>() + box.numel());
        if (scale != 1.0f) {
          boxVector = scaleBox(boxVector, image.cols, image.rows, scale);
        }
        int x1 = static_cast<int>(boxVector[0]);
        int y1 = static_cast<int>(boxVector[1]);
        int x2 = static_cast<int>(boxVector[2]);
        int y2 = static_cast<int>(boxVector[3]);
        int w = x2 - x1;
        int h = y2 - y1;

        // Blur region inside ellipse
        cv::Scalar color;
        if (image.channels() == 3) {
          color = cv::Scalar(255, 255, 255);
        } else {
          color = cv::Scalar(255);
        }

        cv::ellipse(
            mask,
            cv::Point((x1 + x2) / 2, (y1 + y2) / 2),
            cv::Size(w / 2, h / 2),
            0,
            0,
            360,
            color,
            -1);
        // Apply blur effect to the whole image
        cv::Size ksize = cv::Size(image.rows / 8, image.cols / 8);
        cv::Mat blurredImage;
        cv::blur(image(cv::Rect({x1, y1, w, h})), blurredImage, ksize);
        blurredImage.copyTo(
            response(cv::Rect({x1, y1, w, h})), mask(cv::Rect({x1, y1, w, h})));
        blurredImage.release();
      }
    }
    mask.release();
    return response;
  }

  cv::Mat detectAndBlur(
      vrs::utils::PixelFrame* frame,
      const std::string& frameId) {
    // Convert PixelFrame to cv::Mat
    const int width = frame->getWidth();
    const int height = frame->getHeight();
    // Deduce type of the Array (can be either GRAY or RGB)
    const int channels =
        frame->getPixelFormat() == vrs::PixelFormat::RGB8 ? 3 : 1;

    cv::Mat img = cv::Mat(
                      height,
                      width,
                      CV_8UC(channels),
                      static_cast<void*>(frame->getBuffer().data()))
                      .clone();

    // Rotate image if needed
    if (clockwise90Rotation_) {
      cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
    }

    torch::NoGradGuard no_grad;

    // Convert image to tensor
    torch::Tensor imgTensor = torch::from_blob(
        (void*)frame->rdata(), {height, width, channels}, torch::kUInt8);
    // torch::Tensor imgTensor = getImageTensor(frame);
    torch::Tensor imgTensorFloat = imgTensor.to(torch::kFloat);

    // If you need to move to GPU
    torch::Tensor imgTensorFloatOnDevice = imgTensorFloat.to(device_);

    torch::Tensor imgTensorFloatOnDevicePostRotation;
    // rotate the image clockwise
    if (clockwise90Rotation_) {
      imgTensorFloatOnDevicePostRotation =
          torch::rot90(imgTensorFloatOnDevice, -1);
    } else {
      imgTensorFloatOnDevicePostRotation = imgTensorFloatOnDevice;
    }
    // convert from HWC to CHW
    torch::Tensor imgTensorFloatOnDevicePostRotationCHW =
        imgTensorFloatOnDevicePostRotation.permute({2, 0, 1});

    // Create input tensor for model inference
    std::vector<torch::jit::IValue> inputs = {
        imgTensorFloatOnDevicePostRotationCHW};

    // Create output vector to store results
    std::vector<torch::Tensor> boundingBoxes;

    cv::Mat finalImage;

    torch::Tensor faceBoundingBoxes;
    torch::Tensor licensePlateBoundingBoxes;

    // Begin making detections
    // use face model to find faces
    if (faceModel_) {
      c10::intrusive_ptr<c10::ivalue::Tuple> faceDetections =
          faceModel_->forward(inputs)
              .toTuple(); // returns boxes, labels, scores, dims
      faceBoundingBoxes =
          filterDetections(faceDetections, faceModelConfidenceThreshold_);
      int totalFaceDetectionsForCurrentFrame = faceBoundingBoxes.sizes()[0];
      stats_[frameId]["faces"] += totalFaceDetectionsForCurrentFrame;
      if (faceBoundingBoxes.sizes()[0] > 0) {
        boundingBoxes.push_back(faceBoundingBoxes);
      }
      faceDetections.reset();
    }

    // use LP model to find LP
    if (licensePlateModel_) {
      c10::intrusive_ptr<c10::ivalue::Tuple> licensePlateDetections =
          licensePlateModel_->forward(inputs)
              .toTuple(); // returns boxes, labels, scores, dims
      licensePlateBoundingBoxes = filterDetections(
          licensePlateDetections, licensePlateModelConfidenceThreshold_);
      int totaLlicensePlateDetectionsForCurrentFrame =
          licensePlateBoundingBoxes.sizes()[0];
      stats_[frameId]["licensePlate"] +=
          totaLlicensePlateDetectionsForCurrentFrame;
      if (licensePlateBoundingBoxes.sizes()[0] > 0) {
        boundingBoxes.push_back(licensePlateBoundingBoxes);
      }
      licensePlateDetections.reset();
    }

    if (!boundingBoxes.empty()) {
      // Blur the image
      finalImage = blurImage(img, boundingBoxes, scaleFactorDetections_);

      // Rotate image back if needed
      if (clockwise90Rotation_) {
        cv::rotate(finalImage, finalImage, cv::ROTATE_90_COUNTERCLOCKWISE);
      }
      // Force Cleanup
      boundingBoxes.clear();
    }
    // Force Cleanup
    inputs.clear();
    imgTensor.reset();
    imgTensorFloat.reset();
    imgTensorFloatOnDevice.reset();
    imgTensorFloatOnDevicePostRotation.reset();
    imgTensorFloatOnDevicePostRotationCHW.reset();
    faceBoundingBoxes.reset();
    licensePlateBoundingBoxes.reset();
    img.release();
    return finalImage;
  }

  bool operator()(
      double timestamp,
      const vrs::StreamId& streamId,
      vrs::utils::PixelFrame* frame) override {
    // Handle the case where we have no image data
    if (!frame) {
      return false;
    }

    cv::Mat blurredImage;
    // If not Eye Tracking image
    if (streamId.getNumericName().find("214") != std::string::npos ||
        streamId.getNumericName().find("1201") != std::string::npos) {
      // Get predictions and blur
      std::string frameId =
          streamId.getNumericName() + "_" + std::to_string(timestamp);
      stats_[frameId]["faces"] = 0;
      stats_[frameId]["licensePlate"] = 0;
      blurredImage = detectAndBlur(frame, frameId);
    }
    // Copy back results into the frame
    if (!blurredImage.empty()) {
      // RGB
      if (streamId.getNumericName().find("214") != std::string::npos) {
        std::memcpy(
            frame->wdata(),
            blurredImage.data,
            frame->getWidth() * frame->getStride());
      }
      // Gray
      else if (streamId.getNumericName().find("1201") != std::string::npos) {
        std::memcpy(
            frame->wdata(),
            blurredImage.data,
            frame->getWidth() * frame->getHeight());
      }
    }
    blurredImage.release();
    c10::cuda::CUDACachingAllocator::emptyCache();
    return true;
  }

  std::string logStatistics() const {
    std::string statsString;
    int totalFrames = 0;
    int totalRGBFramesWithFaces = 0;
    int totalRGBFaces = 0;
    int totalSLAMFramesWithFaces = 0;
    int totalSLAMFaces = 0;
    int totalRGBFramesWithLicensePlate = 0;
    int totalRGBLicensePlate = 0;
    int totalSLAMFramesWithLicensePlate = 0;
    int totalSLAMLicensePlate = 0;

    for (const auto& outer : stats_) {
      const std::string& frameId = outer.first;
      const std::unordered_map<std::string, int>& categoryBoxCountMapping =
          outer.second;

      // Do something with the outer key and inner map
      for (const auto& innerPair : categoryBoxCountMapping) {
        const std::string& category = innerPair.first;
        int boxCount = innerPair.second;

        if (boxCount > 0) {
          if (category == "faces") {
            if (frameId.find("214") != std::string::npos) {
              totalRGBFramesWithFaces++;
              totalRGBFaces += boxCount;
            } else if (frameId.find("1201") != std::string::npos) {
              totalSLAMFramesWithFaces++;
              totalSLAMFaces += boxCount;
            }
          }
          if (category == "licensePlate") {
            if (frameId.find("214") != std::string::npos) {
              totalRGBFramesWithLicensePlate++;
              totalRGBLicensePlate += boxCount;
            } else if (frameId.find("1201") != std::string::npos) {
              totalSLAMFramesWithLicensePlate++;
              totalSLAMLicensePlate += boxCount;
            }
          }
        }
      }
      totalFrames++;
    }

    std::ostringstream summary;
    summary << " ----------------" << "\n|    Summary     |"
            << "\n ----------------" << "\nTotal frames: " << totalFrames
            << "\n Faces:" << "\n RGB - Total detected frame: "
            << totalRGBFramesWithFaces
            << "\n RGB - Total detections: " << totalRGBFaces
            << "\n SLAM - Total detected frame: " << totalSLAMFramesWithFaces
            << "\n SLAM - Total detections: " << totalSLAMFaces
            << "\n License Plates:" << "\n RGB - Total detected frame: "
            << totalRGBFramesWithLicensePlate
            << "\n RGB - Total detections: " << totalRGBLicensePlate
            << "\n SLAM - Total detected frame: "
            << totalSLAMFramesWithLicensePlate
            << "\n SLAM - Total detections: " << totalSLAMLicensePlate;
    return summary.str();
  }
};

} // namespace EgoBlur
