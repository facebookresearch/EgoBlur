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

#include <cstdlib>
#include <memory>
#include <string>

#include <fmt/core.h>
#include <vrs/utils/FilterCopy.h> // @manual
#include <vrs/utils/RecordFileInfo.h> // @manual

#include <projectaria_tools/tools/samples/vrs_mutation/ImageMutationFilterCopier.h> // @manual
#include "EgoBlurImageMutator.h"

#include <CLI/CLI.hpp>

int main(int argc, const char* argv[]) {
  // std::string mutationType;
  std::string vrsPathIn;
  std::string vrsPathOut;
  std::string vrsExportPath;
  std::string faceModelPath;
  std::string licensePlateModelPath;
  float faceModelConfidenceThreshold;
  float licensePlateModelConfidenceThreshold;
  float scaleFactorDetections;
  float nmsThreshold;
  bool useGPU = false;

  CLI::App app{
      "VRS file Mutation example by using VRS Copy + Filter mechanism"};

  app.add_option("-i,--in", vrsPathIn, "VRS input")->required();
  app.add_option("-o,--out", vrsPathOut, "VRS output")->required();
  app.add_option("-f, --faceModelPath", faceModelPath, "Face model path");
  app.add_option(
         "--face-model-confidence-threshold",
         faceModelConfidenceThreshold,
         "Face model confidence threshold")
      ->default_val(0.1);
  app.add_option(
      "-l, --licensePlateModelPath",
      licensePlateModelPath,
      "License Plate model path");
  app.add_option(
         "--license-plate-model-confidence-threshold",
         licensePlateModelConfidenceThreshold,
         "License plate model confidence threshold")
      ->default_val(0.1);
  app.add_option(
         "--scale-factor-detections",
         scaleFactorDetections,
         "scale factor for scaling detections in dimensions")
      ->default_val(1.15);
  app.add_option(
         "--nms-threshold",
         nmsThreshold,
         "NMS threshold for filtering overlapping detections")
      ->default_val(0.3);
  app.add_flag("--use-gpu", useGPU, "Use GPU for inference");
  app.add_option("-e,--exportPath", vrsExportPath, "VRS export output path");

  CLI11_PARSE(app, argc, argv);

  if (vrsPathIn == vrsPathOut) {
    std::cerr << " <VRS_IN> <VRS_OUT> paths must be different." << std::endl;
    return EXIT_FAILURE;
  }

  vrs::utils::FilteredFileReader filteredReader;
  // Initialize VRS Reader and filters
  filteredReader.setSource(vrsPathIn);
  filteredReader.openFile();
  filteredReader.applyFilters({});

  // Configure Copy Filter and initialize the copy
  const std::string targetPath = vrsPathOut;
  vrs::utils::CopyOptions copyOptions;
  copyOptions.setCompressionPreset(vrs::CompressionPreset::Default);

  // Functor to perform image processing(blurring PII faces/license plates)
  try {
    std::shared_ptr<vrs::utils::UserDefinedImageMutator> imageMutator;
    if (setenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "1", 1) == 0) {
      // See github issue https://github.com/pytorch/pytorch/issues/29893 for
      // details
      std::cout << "Successfully Set ONEDNN_PRIMITIVE_CACHE_CAPACITY to 1"
                << std::endl;
    }
    if (setenv("TORCH_CUDNN_V8_API_DISABLED", "1", 1) == 0) {
      std::cout << "Successfully Set TORCH_CUDNN_V8_API_DISABLED to 1"
                << std::endl;
    }
    imageMutator = std::make_shared<EgoBlur::EgoBlurImageMutator>(
        faceModelPath,
        faceModelConfidenceThreshold,
        licensePlateModelPath,
        licensePlateModelConfidenceThreshold,
        scaleFactorDetections,
        nmsThreshold,
        useGPU);

    auto copyMakeStreamFilterFunction =
        [&imageMutator](
            vrs::RecordFileReader& fileReader,
            vrs::RecordFileWriter& fileWriter,
            vrs::StreamId streamId,
            const vrs::utils::CopyOptions& copyOptions)
        -> std::unique_ptr<vrs::utils::RecordFilterCopier> {
      auto imageMutatorFilter =
          std::make_unique<vrs::utils::ImageMutationFilter>(
              fileReader,
              fileWriter,
              streamId,
              copyOptions,
              imageMutator.get());
      return imageMutatorFilter;
    };

    const int statusCode = filterCopy(
        filteredReader, targetPath, copyOptions, copyMakeStreamFilterFunction);
    auto* const egoBlurMutator =
        dynamic_cast<EgoBlur::EgoBlurImageMutator*>(imageMutator.get());
    std::cout << egoBlurMutator->logStatistics() << std::endl;
    return statusCode;
  } catch (const std::exception& ex) {
    std::cerr << "Error while applying EGOBLUR mutation : " << " to : "
              << vrsPathIn << "\nError :\n"
              << ex.what() << std::endl;
    return EXIT_FAILURE;
  }
}
