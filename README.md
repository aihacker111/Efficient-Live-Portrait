# Efficient-Live-Portrait

## Introduction
This repo is the optimize task by converted to ONNX models for [LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control](https://github.com/KwaiVGI/LivePortrait).
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.

Also we adding feature: Real-Time demo with ONNX models

## 🔥 Getting Started
### 1. Clone the code and prepare the environment
```bash
git clone https://github.com/aihacker111/Efficient-Live-Portrait
# create env using conda
conda create -n ELivePortrait python==3.10.14
conda activate ELivePortrait
# install dependencies with pip
pip install -r requirements.txt
```

**Note:** make sure your system has [FFmpeg](https://ffmpeg.org/) installed!

### 2. Download pretrained weights

The pretrained weights is also automatic downloading
You don't need to download and put model into sources code
```text
pretrained_weights
|
├── landmarks
│   └── models
│       └── buffalo_l
│       |   ├── 2d106det.onnx
│       |    └── det_10g.onnx
|       └── landmark.onnx
└── live_portrait
      |
      ├── appearance_feature_extractor.onnx
      ├── motion_extractor.onnx
      ├── spade_generator.onnx
      └── warping.onnx
      ├── stitching_retargeting.onnx
      └── stitching_retargeting_eye.onnx
      └── stitching_retargeting_lip.onnx
      

```
### 3. Inference and Real-time Demo 🚀
#### Fast hands-on
```bash
python run_live_portrait.py -v 'path/to/your/video/driving/or/webcam/id' -i 'path/to/your/image/want/to/animation' -r '/use/it/when/you/want/to/run/real-time/'
```
### 5. Inference speed evaluation 🚀🚀🚀

We'll release it soon