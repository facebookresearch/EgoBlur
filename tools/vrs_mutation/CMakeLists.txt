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

cmake_minimum_required(VERSION 3.12)
project(ego_blur
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PROJECTARIA_TOOLS_BUILD_TOOLS ON CACHE BOOL "")


include(FetchContent)
FetchContent_Declare(
    projectaria_tools
    GIT_REPOSITORY https://github.com/facebookresearch/projectaria_tools.git
    GIT_TAG origin/main
    SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/projectaria_tools-src/projectaria_tools"
)
FetchContent_MakeAvailable(projectaria_tools)
include_directories("${CMAKE_BINARY_DIR}/_deps/projectaria_tools-src")



find_package(TorchVision REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED)


add_executable( ego_blur_vrs_mutation
    EgoBlurImageMutator.h
    main.cpp
)
target_link_libraries(ego_blur_vrs_mutation
    vrs_image_mutation_interface
    TorchVision::TorchVision
    CLI11::CLI11
    "${TORCH_LIBRARIES}"
    ${OpenCV_LIBS})
