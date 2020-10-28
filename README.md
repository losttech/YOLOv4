# YOLOv4

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

*NOTICE: This is a port of https://github.com/SoloSynth1/tensorflow-yolov4

YOLOv4 Implemented in Tensorflow 1.15

### Prerequisites
[![LostTech.TensorFlow](https://img.shields.io/nuget/v/LostTech.TensorFlow.svg?label=nuget:%20LostTech.TensorFlow)](https://www.nuget.org/packages/LostTech.TensorFlow)

### Performance
<p align="center"><img src="data/performance.png" width="640"\></p>

### Demo

TBD

#### Output

##### Yolov4 original weight
<p align="center"><img src="data/result.png" width="640"\></p>

##### Yolov4 tflite int8
<p align="center"><img src="data/result-int8.png" width="640"\></p>

### Convert to ONNX

TBD

### Evaluate on COCO 2017 Dataset

TBD

# evaluate yolov4 model

TBD

#### mAP50 on COCO 2017 Dataset

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3      | 55.43   |         |         |
| YoloV4      | 61.96   | 57.33   |         |

### Benchmark

TBD

#### Tesla P100

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3 FPS  | 40.6    | 49.4    | 61.3    |
| YoloV4 FPS  | 33.4    | 41.7    | 50.0    |

#### Tesla K80

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3 FPS  | 10.8    | 12.9    | 17.6    |
| YoloV4 FPS  | 9.6     | 11.7    | 16.0    |

#### Tesla T4

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3 FPS  | 27.6    | 32.3    | 45.1    |
| YoloV4 FPS  | 24.0    | 30.3    | 40.1    |

#### Tesla P4

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3 FPS  | 20.2    | 24.2    | 31.2    |
| YoloV4 FPS  | 16.2    | 20.2    | 26.5    |

#### Macbook Pro 15 (2.3GHz i7)

| Detection   | 512x512 | 416x416 | 320x320 |
|-------------|---------|---------|---------|
| YoloV3 FPS  |         |         |         |
| YoloV4 FPS  |         |         |         |

### Traning your own model

Sample training code available at [samples/TrainV4](samples/TrainV4)

### References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [darknet](https://github.com/AlexeyAB/darknet)
  
   My project is inspired by these previous fantastic YOLOv3 implementations:
  * [Yolov3 tensorflow](https://github.com/YunYang1994/tensorflow-yolov3)
  * [Yolov3 tf2](https://github.com/zzh8829/yolov3-tf2)