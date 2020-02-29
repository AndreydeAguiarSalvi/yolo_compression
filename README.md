# Introduction

This directory contains my PyTorch YOLOv3 study about pruning. **It is freely available for redistribution under the GPL-3.0 license**. 
This work was based on the code of [YOLOv3 Ultralytics](https://github.com/ultralytics/yolov3)


# Requirements

Python 3.7 or later with all of the `pip install -U -r requirements.txt` packages including:
- `numpy = 1.17 (version 1.18 raises bugs on COCOAPI)`
- `torch >= 1.3`
- `opencv-python`
- `Pillow`


# Training

**Start Training:** `python3 train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`. Each epoch trains on 117,263 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set.

**Resume Training:** `python3 train.py --resume` to resume training from `weights/last.pt`.

**Plot Training:** `from utils import utils; utils.plot_results()` plots training results from `coco_16img.data`, `coco_64img.data`, 2 example datasets available in the `data/` folder, which train and test on the first 16 and 64 images of the COCO2014-trainval dataset.


## Image Augmentation

`datasets.py` applies random OpenCV-powered (https://opencv.org/) augmentation to the input images in accordance with the following specifications. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images. 416 x 416 examples pictured below.

Augmentation | Description
--- | ---
Translation | +/- 10% (vertical and horizontal)
Rotation | +/- 5 degrees
Shear | +/- 2 degrees (vertical and horizontal)
Scale | +/- 10%
Reflection | 50% probability (horizontal-only)
H**S**V Saturation | +/- 50%
HS**V** Intensity | +/- 50%


## Speed

https://cloud.google.com/deep-learning-vm/  
**Machine type:** preemptible [n1-standard-16](https://cloud.google.com/compute/docs/machine-types) (16 vCPUs, 60 GB memory)   
**CPU platform:** Intel Skylake  
**GPUs:** K80 ($0.20/hr), T4 ($0.35/hr), V100 ($0.83/hr) CUDA with [Nvidia Apex](https://github.com/NVIDIA/apex) FP16/32  
**HDD:** 1 TB SSD  
**Dataset:** COCO train 2014 (117,263 images)  
**Model:** `yolov3-spp.cfg`  
**Command:**  `python3 train.py --img 416 --batch 32 --accum 2`

GPU |n| `--batch --accum` | img/s | epoch<br>time | epoch<br>cost
--- |--- |--- |--- |--- |---
K80    |1| 32 x 2 | 11  | 175 min  | $0.58
T4     |1<br>2| 32 x 2<br>64 x 1 | 41<br>61 | 48 min<br>32 min | $0.28<br>$0.36
V100   |1<br>2| 32 x 2<br>64 x 1 | 122<br>**178** | 16 min<br>**11 min** | **$0.23**<br>$0.31
2080Ti |1<br>2| 32 x 2<br>64 x 1 | 81<br>140 | 24 min<br>14 min | -<br>-

# Inference

`detect.py` runs inference on any sources:

```bash
python3 detect.py --source ...
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

To run a specific models:

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.weights`  

**YOLOv3-tiny:** `python3 detect.py --cfg cfg/yolov3-tiny.cfg --weights yolov3-tiny.weights`  

**YOLOv3-SPP:** `python3 detect.py --cfg cfg/yolov3-spp.cfg --weights yolov3-spp.weights`  


# Pretrained Weights

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)

## Darknet Conversion

```bash
$ git clone https://github.com/ultralytics/yolov3 && cd yolov3

# convert darknet cfg/weights to pytorch model
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'converted.pt'

# convert cfg/pytorch model to darknet weights
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.pt')"
Success: converted 'weights/yolov3-spp.pt' to 'converted.weights'
```

# mAP

```bash
$ python3 test.py --cfg yolov3-spp.cfg --weights yolov3-spp-ultralytics.pt
```

- mAP@0.5 run at `--iou-thr 0.5`, mAP@0.5...0.95 run at `--iou-thr 0.7`
- Darknet results: https://arxiv.org/abs/1804.02767

<i></i>                      |Size |COCO mAP<br>@0.5...0.95 |COCO mAP<br>@0.5 
---                          | ---         | ---         | ---
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |320 |14.0<br>28.7<br>30.5<br>**36.6** |29.1<br>51.8<br>52.3<br>**56.0**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |416 |16.0<br>31.2<br>33.9<br>**40.4** |33.0<br>55.4<br>56.9<br>**60.2**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |512 |16.6<br>32.7<br>35.6<br>**41.6** |34.9<br>57.7<br>59.5<br>**61.7**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |608 |16.6<br>33.1<br>37.0<br>**42.1** |35.4<br>58.2<br>60.7<br>**61.7**

```bash
$ python3 test.py --cfg yolov3-spp.cfg --weights yolov3-spp-ultralytics.pt --img 608
Namespace(batch_size=32, cfg='yolov3-spp', conf_thres=0.001, data='data/coco2014.data', device='', img_size=608, iou_thres=0.6, save_json=True, single_cls=False, task='test', weights='last82.pt')
Using CUDA device0 _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', total_memory=16280MB)
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 157/157 [03:12<00:00,  1.50it/s]
                 all     5e+03  3.51e+04    0.0573     0.871     0.611     0.106
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.618
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.462
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.534
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.341
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.557
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.606
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.735
```

# Reproduce Our Results

This command trains `yolov3-spp.cfg` from scratch to our mAP above. Training takes about one week on a 2080Ti.
```bash
$ python3 train.py --weights '' --cfg yolov3-spp.cfg --epochs 273 --batch 16 --accum 4 --multi_scale
```
<img src="https://user-images.githubusercontent.com/26833433/70661588-76bbca00-1c19-11ea-86f9-23350d8c3193.png" width="900">

# Reproduce Our Environment

To access an up-to-date working environment (with all dependencies including CUDA/CUDNN, Python and PyTorch preinstalled), consider a:

- **GCP** Deep Learning VM with $300 free credit offer: See our [GCP Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart) 
- **Google Colab Notebook** with 12 hours of free GPU time: [Google Colab Notebook](https://colab.research.google.com/drive/1G8T-VFxQkjDe4idzN8F-hbIBqkkkQnxw)
- **Docker Image** from https://hub.docker.com/r/ultralytics/yolov3. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/Docker-Quickstart) 
# Citation

[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)

# Contact

**Issues should be raised directly in the repository.** For additional questions or comments please email Glenn Jocher at glenn.jocher@ultralytics.com or visit us at https://contact.ultralytics.com.
