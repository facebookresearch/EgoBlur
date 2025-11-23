# EgoBlur Demo
This repository contains demo of [EgoBlur models](https://www.projectaria.com/tools/egoblur) with visualizations.


## Installation

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

## Getting Started
First download the zipped models from given links. Then the models can be used as input/s to CLI.

| Model | Download link |
| -------- | -------- |
| ego_blur_face_gen1 | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |
| ego_blur_lp_gen1 | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |
| ego_blur_face_gen2 | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |
| ego_blur_lp_gen2 | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |


### CLI options

A brief description of CLI args:

`--face_model_path` use this argument to provide absolute EgoBlur face model file path. You MUST provide either `--face_model_path` or `--lp_model_path` or both. If none is provided code will throw a `ValueError`.

`--face_model_score_threshold` use this argument to provide face model score threshold to filter out low confidence face detections. The values must be between 0.0 and 1.0, if not provided this defaults to 0.1.

`--lp_model_path` use this argument to provide absolute EgoBlur license plate file path. You MUST provide either `--face_model_path` or `--lp_model_path` or both. If none is provided code will throw a `ValueError`.

`--lp_model_score_threshold` use this argument to provide license plate model score threshold to filter out low confidence license plate detections. The values must be between 0.0 and 1.0, if not provided this defaults to 0.1.

`--nms_iou_threshold` use this argument to provide NMS iou threshold to filter out low confidence overlapping boxes. The values must be between 0.0 and 1.0, if not provided this defaults to 0.3.

`--scale_factor_detections` use this argument to provide scale detections by the given factor to allow blurring more area. The values can only be positive real numbers eg: 0.9(values < 1) would mean scaling DOWN the predicted blurred region by 10%, whereas as 1.1(values > 1) would mean scaling UP the predicted blurred region by 10%.

`--input_image_path` use this argument to provide absolute path for the given image on which we want to make detections and perform blurring. You MUST provide either `--input_image_path` or `--input_video_path` or both. If none is provided code will throw a `ValueError`.

`--output_image_path` use this argument to provide absolute path where we want to store the blurred image. You MUST provide `--output_image_path` with `--input_image_path` otherwise code will throw `ValueError`.

`--input_video_path` use this argument to provide absolute path for the given video on which we want to make detections and perform blurring. You MUST provide either `--input_image_path` or `--input_video_path` or both. If none is provided code will throw a `ValueError`.

`--output_video_path` use this argument to provide absolute path where we want to store the blurred video. You MUST provide `--output_video_path` with `--input_video_path` otherwise code will throw `ValueError`.

`--output_video_fps` use this argument to provide FPS for the output video. The values must be positive integers, if not provided this defaults to 30.



### CLI command examples
Download this repository to access the built-in demo assets. The samples are organized per generation:

- Gen2 assets live under `${EGOBLUR_REPO}/gen2/demo_assets/`
- Gen1 assets live under `${EGOBLUR_REPO}/gen1/demo_assets/`

Update the snippets below with absolute paths that match your system, for example:

```
export EGOBLUR_REPO=/absolute/path/to/ego_blur_public_internal
export EGOBLUR_MODELS=/home/${USER}/ego_blur_assets
```

Extract the downloaded model files into `${EGOBLUR_MODELS}` (for example, `${EGOBLUR_MODELS}/ego_blur_face_gen2.jit`). After installing the package, run:

```
egoblur-gen2 --help
```

#### Face blurring (image)
```
egoblur-gen2 \
  --face_model_path ${EGOBLUR_MODELS}/ego_blur_face_gen2.jit \
  --input_image_path ${EGOBLUR_REPO}/gen2/demo_assets/test_face_image.png \
  --output_image_path ${EGOBLUR_REPO}/gen2/demo_assets/output/test_face_image_output.png
```

#### License plate blurring (image)
```
egoblur-gen2 \
  --lp_model_path ${EGOBLUR_MODELS}/ego_blur_lp_gen2.jit \
  --input_image_path ${EGOBLUR_REPO}/gen2/demo_assets/test_lp_image.png \
  --output_image_path ${EGOBLUR_REPO}/gen2/demo_assets/output/test_lp_image_output.png
```

#### Combined face + license plate (image + video)
```
egoblur-gen2 \
  --face_model_path ${EGOBLUR_MODELS}/ego_blur_face_gen2.jit \
  --lp_model_path ${EGOBLUR_MODELS}/ego_blur_lp_gen2.jit \
  --input_image_path ${EGOBLUR_REPO}/gen2/demo_assets/test_face_image.png \
  --output_image_path ${EGOBLUR_REPO}/gen2/demo_assets/output/test_face_image_output.png \
  --input_video_path ${EGOBLUR_REPO}/gen2/demo_assets/test_face_video.mp4 \
  --output_video_path ${EGOBLUR_REPO}/gen2/demo_assets/output/test_face_video_output.mp4 \
  --face_model_score_threshold 0.9 \
  --lp_model_score_threshold 0.9 \
  --nms_iou_threshold 0.3 \
  --scale_factor_detections 1.0 \
  --output_video_fps 20
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

The CLI arguments are otherwise identical between `egoblur-gen1` and `egoblur-gen2`.

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

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
