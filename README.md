# EgoBlur Demo
This repository contains demo of [EgoBlur models](https://www.projectaria.com/tools/egoblur) with visualizations.


## Installation

This code requires `conda>=23.1.0` to install dependencies and create a virtual environment to execute the code in. Please follow the instructions [here](https://docs.anaconda.com/free/anaconda/install/index.html) to install Anaconda for your machine.

We list our dependencies in `environment.yaml` file. To install the dependencies and create the env run:
```
conda env create --file=environment.yaml

# After installation, check pytorch.
conda activate ego_blur
python
>>> import torch
>>> torch.__version__
'1.12.1'
>>> torch.cuda.is_available()
True
```

Please note that this code can run on both CPU and GPU but installing both PyTorch and TorchVision with CUDA support is strongly recommended.

## Getting Started
First download the zipped models from given links. Then the models can be used as input/s to CLI.

| Model | Download link |
| -------- | -------- |
| ego_blur_face | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |
| ego_blur_lp | [ego_blur_website](https://www.projectaria.com/tools/egoblur) |


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



### CLI command example
Download the git repo locally and run following commands.
Please note that these commands assumes that you have a created a folder `/home/${USER}/ego_blur_assets/` where you have extracted the zipped models and have test image in the form of `test_image.jpg` and a test video in the form of `test_video.mp4`.

```
conda activate ego_blur
```

#### demo command for face blurring on the demo_assets image

```
python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --input_image_path demo_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg
```


#### demo command for face blurring on an image using default arguments

```
python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg
```


#### demo command for face blurring on an image
```
python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg --face_model_score_threshold 0.9 --nms_iou_threshold 0.3 --scale_factor_detections 1.15
```

#### demo command for license plate blurring on an image
```
python script/demo_ego_blur.py --lp_model_path /home/${USER}/ego_blur_assets/ego_blur_lp.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg --lp_model_score_threshold 0.9 --nms_iou_threshold 0.3 --scale_factor_detections 1
```

#### demo command for face blurring and license plate blurring on an input image and video
```
python script/demo_ego_blur.py --face_model_path /home/${USER}/ego_blur_assets/ego_blur_face.jit --lp_model_path /home/${USER}/ego_blur_assets/ego_blur_lp.jit --input_image_path /home/${USER}/ego_blur_assets/test_image.jpg --output_image_path /home/${USER}/ego_blur_assets/test_image_output.jpg  --input_video_path /home/${USER}/ego_blur_assets/test_video.mp4 --output_video_path /home/${USER}/ego_blur_assets/test_video_output.mp4 --face_model_score_threshold 0.9 --lp_model_score_threshold 0.9 --nms_iou_threshold 0.3 --scale_factor_detections 1 --output_video_fps 20
```

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
