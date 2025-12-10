# EgoBlur Demo

This repository contains a demo of
[EgoBlur models](https://www.projectaria.com/tools/egoblur) with visualizations.

## Overview

EgoBlur is an open-source AI model from Meta designed to preserve privacy by
detecting and blurring faces and license plates in images and videos. We provide
Gen1 models for data captured with Aria Gen1 devices and Gen2 models for data
captured with Aria Gen2 devices.

Gen1:

- egoblur-gen1 command line tool — Python tool for PNG, JPEG, or MP4 files
- EgoBlur VRS Utilities — C++ tool for Aria Gen1 VRS files
  - See the [EgoBlur VRS Utilities README](./gen1/tools/README.md) for details.
- See
  [EgoBlur Gen1 wiki](https://facebookresearch.github.io/projectaria_tools/docs/open_models/egoblur)
  for Gen1 models' details.

Gen2 - ✨new✨:

- egoblur-gen2 command line tool — Python tool for PNG, JPEG, or MP4 files
- EgoBlur VRS Utilities — C++ tool for Aria Gen2 VRS files (coming soon)
- See
  [EgoBlur Gen2 wiki](https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/models/egoblur)
  for Gen2 models' details.

<img src="./assets/gen2_result.jpeg" alt="EgoBlur Gen2 Demo" style="width: 50%;">

## Installation

### Supported environments

Gen1 and Gen2 python command line tool:

- Python version: 3.10-3.12
- Platform: Fedora 43, Ubuntu 20.04, Ubuntu 22.04, Ubuntu 24.04, MacOS-14,
  MacOS-15
- CUDA toolkit (optional): 12.8

Gen1 C++ VRS Utilities:

- Ubuntu 22.04
- libtorch 2.1
- CUDA toolkit 12.1

### Installation from PyPI

1. (Optional) Create and activate a fresh virtual environment:
   ```
   mkdir -p ~/venvs
   python3 -m venv ~/venvs/egoblur
   source ~/venvs/egoblur/bin/activate
   pip install --upgrade pip
   ```
2. Install EgoBlur directly from PyPI:

   ```
   pip install egoblur
   ```

3. Download the repo:
   ```
   git clone https://github.com/facebookresearch/EgoBlur.git
   ```

## Getting Started

First, download the zipped models from the links provided. You can then pass
these models as inputs to the CLI.

| Model              | Download link                                                 |
| ------------------ | ------------------------------------------------------------- |
| ego_blur_face_gen1 | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |
| ego_blur_lp_gen1   | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |
| ego_blur_face_gen2 | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |
| ego_blur_lp_gen2   | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |

### CLI options

The CLI arguments are mostly identical between `egoblur-gen1` and
`egoblur-gen2`, with two differences:

- Gen2 adds optional `--camera_name` for camera-specific defaults.
- Gen2 preserves the input video FPS; Gen1 retains `--output_video_fps` for
  manually setting the output FPS.

A brief description of CLI args:

`--camera_name` **(Gen2 only)** use this argument to specify the camera type for
automatic threshold selection. Valid options are: `slam-front-left`,
`slam-front-right`, `slam-side-left`, `slam-side-right`, `camera-rgb`. See the
stream id to camera label in
[project aria wiki](https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/vrs/streamid-label-mapper).
When specified, camera-specific default thresholds will be used for both face
and license plate detection (see Camera-Specific Thresholds section below). This
is optional.

`--lp_model_path` use this argument to provide the absolute EgoBlur license
plate model file path. file path. You MUST provide either `--face_model_path` or
`--lp_model_path` or both. If none is provided code will throw a `ValueError`.

`--face_model_score_threshold` use this argument to provide face model score
threshold to filter out low confidence face detections. The values must be
between 0.0 and 1.0. If not provided and `--camera_name` is specified (Gen2
only), camera-specific defaults are used. Otherwise set to default threshold for
camera rgb.

`--lp_model_path` use this argument to provide absolute EgoBlur license plate
file path. You MUST provide either `--face_model_path` or `--lp_model_path` or
both. If none is provided code will throw a `ValueError`.

`--lp_model_score_threshold` use this argument to provide license plate model
score threshold to filter out low confidence license plate detections. The
values must be between 0.0 and 1.0. If not provided and `--camera_name` is
specified (Gen2 only), camera-specific defaults are used. Otherwise set to
default threshold for camera rgb.

`--nms_iou_threshold` use this argument to provide the NMS IoU threshold to
filter out low confidence overlapping boxes. The values must be between 0.0 and
1.0, if not provided this defaults to 0.3.

`--scale_factor_detections` use this argument to scale detections by the given
factor to allow blurring more area. The values can only be positive real
numbers, e.g., 0.9 (values < 1) would mean scaling DOWN the predicted blurred
region by 10%, whereas 1.1 (values > 1) would mean scaling UP the predicted
blurred region by 10%.

`--input_image_path` use this argument to provide the absolute path for the
given image image on which we want to make detections and perform blurring. You
MUST provide either `--input_image_path` or `--input_video_path` or both. If
none is provided code will throw a `ValueError`.

`--output_image_path` use this argument to provide the absolute path where we
want to store the blurred image. to store the blurred image. You MUST provide
`--output_image_path` with `--input_image_path` otherwise code will throw
`ValueError`.

`--input_video_path` use this argument to provide the absolute path for the
given video video on which we want to make detections and perform blurring. You
MUST provide either `--input_image_path` or `--input_video_path` or both. If
none is provided code will throw a `ValueError`.

`--output_video_path` use this argument to provide the absolute path where we
want to store the blurred video. to store the blurred video. You MUST provide
`--output_video_path` with `--input_video_path` otherwise code will throw
`ValueError`.

Video FPS handling: the Gen2 script preserves the input video's FPS when writing
the blurred output. If the input FPS metadata is missing or invalid, the script
raises a `ValueError`; provide a video file with a valid, fixed FPS.

`--output_video_fps` **(Gen1 only)** use this argument to provide the FPS for
the output video. The values must be positive integers; if not provided this
defaults to 30.

### CLI command examples

Download this repository to access the built-in demo assets. The samples are
organized per generation:

- Gen2 assets live under `${EGOBLUR_REPO}/gen2/demo_assets/`
- Gen1 assets live under `${EGOBLUR_REPO}/gen1/demo_assets/`

Update the snippets below with absolute paths that match your system, for
example:

```
export EGOBLUR_REPO=/absolute/path/to/ego_blur
export EGOBLUR_MODELS=/home/${USER}/ego_blur_assets
```

Extract the downloaded model files into `${EGOBLUR_MODELS}` (for example,
`${EGOBLUR_MODELS}/ego_blur_face_gen2.jit`). After installing the package, run:

```
egoblur-gen2 --help
```

#### Face blurring (image)

```
egoblur-gen2 \
  --camera_name camera-rgb \
  --face_model_path ${EGOBLUR_MODELS}/ego_blur_face_gen2.jit \
  --input_image_path ${EGOBLUR_REPO}/gen2/demo_assets/test_face_image.png \
  --output_image_path ${EGOBLUR_REPO}/gen2/demo_assets/output/test_face_image_output.png
```

#### Face blurring (video)

```
egoblur-gen2 \
  --camera_name camera-rgb \
  --face_model_path ${EGOBLUR_MODELS}/ego_blur_face_gen2.jit \
  --input_video_path ${EGOBLUR_REPO}/gen2/demo_assets/test_face_video.mp4 \
  --output_video_path ${EGOBLUR_REPO}/gen2/demo_assets/output/test_face_video_output.mp4
```

#### License plate blurring (image)

```
egoblur-gen2 \
  --camera_name camera-rgb \
  --lp_model_path ${EGOBLUR_MODELS}/ego_blur_lp_gen2.jit \
  --input_image_path ${EGOBLUR_REPO}/gen2/demo_assets/test_lp_image.png \
  --output_image_path ${EGOBLUR_REPO}/gen2/demo_assets/output/test_lp_image_output.png
```

#### License plate blurring (video)

```
egoblur-gen2 \
  --camera_name camera-rgb \
  --lp_model_path ${EGOBLUR_MODELS}/ego_blur_lp_gen2.jit \
  --input_video_path ${EGOBLUR_REPO}/gen2/demo_assets/test_lp_video.mp4 \
  --output_video_path ${EGOBLUR_REPO}/gen2/demo_assets/output/test_lp_video_output.mp4
```

Note: Some license plates in output video frames may remain not blurred due to
low detection confidence. This is expected, you can adjust the
`--lp_model_score_threshold` to add low confidence detections.

#### Combined face + license plate (image + video)

```
egoblur-gen2 \
  --camera_name camera-rgb \
  --face_model_path ${EGOBLUR_MODELS}/ego_blur_face_gen2.jit \
  --lp_model_path ${EGOBLUR_MODELS}/ego_blur_lp_gen2.jit \
  --input_image_path ${EGOBLUR_REPO}/gen2/demo_assets/test_lp_image.png \
  --output_image_path ${EGOBLUR_REPO}/gen2/demo_assets/output/test_lp_image_output.png \
  --input_video_path ${EGOBLUR_REPO}/gen2/demo_assets/test_face_video.mp4 \
  --output_video_path ${EGOBLUR_REPO}/gen2/demo_assets/output/test_face_video_output.mp4
```

#### Gen1 demo assets

Gen1 sample media lives in `${EGOBLUR_REPO}/gen1/demo_assets/`. Sample usage:

```
egoblur-gen1 \
  --face_model_path ${EGOBLUR_MODELS}/ego_blur_face_gen1.jit \
  --lp_model_path ${EGOBLUR_MODELS}/ego_blur_lp_gen1.jit \
  --input_image_path ${EGOBLUR_REPO}/gen1/demo_assets/test_image.jpg \
  --output_image_path ${EGOBLUR_REPO}/gen1/demo_assets/test_image_output.jpg \
  --input_video_path ${EGOBLUR_REPO}/gen1/demo_assets/test_video.mp4 \
  --output_video_path ${EGOBLUR_REPO}/gen1/demo_assets/test_video_output.mp4
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md).

## Citing EgoBlur

If you use EgoBlur in your research, please use the following BibTeX entry.

```
@misc{raina2023egoblur,
      title={EgoBlur: Responsible Innovation in Aria},
      author={Nikhil Raina and Guruprasad Somasundaram and Kang Zheng and Sagar Miglani and Steve Saarinen and Jeff Meissner and Mark Schwesinger and Luis Pesqueira and Ishita Prasad and Edward Miller and Prince Gupta and Mingfei Yan and Richard Newcombe and Carl Ren and Omkar M Parkhi},
      year={2023},
      eprint={2308.13093},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
