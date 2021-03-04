# Introduction

This repository contains my master's (ongoing) work on model compression techniques at YOLOv3. **It is freely available for redistribution under the GPL-3.0 license**. 
This repository is based on [YOLOv3 Ultralytics](https://github.com/ultralytics/yolov3).

Currently evaluated approaches:
* Lottery Tickets Hypothesis (Iterative Magnitude based Pruning)
* Continuous Sparsification (Iterative Gradient based Pruning)
* Knowledge Distillation (classical approach)
* Generative Adversarial Network (GAN) based Knowledge Distillation
* Neural Architecture Search (NAS) from MobileNet V3
* NAS from YOLO Nano

# Requirements

Python 3.7 or later with all of the `pip install -U -r requirements.txt` packages including:
- `numpy = 1.19 (version 1.18 raises bugs on COCOAPI)`
- `torch >= 1.7`
- `opencv-python`
- `Pillow`
- [THOP](https://github.com/Lyken17/pytorch-OpCounter) to count the MACs

# Other Details
I am now focused on completing my master's (scheduled for March, 2020). With this task completed, I will bring you the final results of the work and examples of how to run this repository.
Basically, run 
* train.py to perform a normal training,
* prune.py to perform pruning with LTH or CS, depending on the params
* my_kd.py to perform classical KD with YOLOv3 and YOLO Mobile (model of my own) or YOLO Nano
* my_kd_gan.py to perform my adapted GAN based KD
In utils/my_utils.py, you can see the argument parser, to see all the available parameters

# Results
## Pascal VOC 2007 test set
|Model           |Training     |mAP          |Final Params      |MACs                            |Storage (MB) |
|:--------------:|:-----------:|:-----------:|:----------------:|:------------------------------:|:-----------:|
|  YOLOv3-Tiny   |   Default   |0.379 ± 0.003|   8, 713, 766    |        2, 753, 665, 551        |    33.29    |
|     YOLOv3     |   Default   |0.547 ± 0.012|   61, 626, 049   |       32, 829, 119, 167        |   235.44    |
|   YOLO Nano    |   Default   |0.385 ± 0.007|   2, 890, 527    |        2, 082, 423, 381        |    11.38    |
| YOLOv3-Mobile  |   Default   |0.009 ± 0.008|   4, 395, 985    |        1, 419, 864, 487        |    17.59    |
|     YOLOv3     |  LTH Local  |0.549 ± 0.009| 6, 331, 150 ± 1  |     3, 468, 547, 347 ± 278     |   118.26    |
|     YOLOv3     | LTH Global  |**0.561 ± 0.009**| 6, 331, 114 ± 1  |8, 796, 051, 025 ± 225, 877, 824|   118.26    |
|     YOLOv3     |   CS 1 It   |0.442 ± 0.010|740, 072 ± 12, 161|1, 137, 839, 381 ± 44, 191, 983 |11.618 ± 0.23|
|     YOLOv3     |   CS 3 It   |0.316 ± 0.015|**421, 721 ± 3, 544** |  **618, 724, 616 ± 20, 611, 379**  |**5.544 ± 0.07** |
| YOLO Nano<sub>leaky</sub> |  KD fts 79  |0.421 ± 0.007|   2, 890, 527    |        2, 098, 305, 681        |    11.38    |
| YOLO Nano<sub>leaky</sub> |KD fts 36, 61|0.408 ± 0.008|   2, 890, 527    |        2, 098, 305, 681        |    11.38    |
|YOLO Mobile<sub>leaky</sub>|  KD fts 91  |0.253 ± 0.023|   4, 395, 985    |        1, 458, 910, 247        |    17.59    |
|YOLO Mobile<sub>leaky</sub>|KD fts 36, 91|0.244 ± 0.010|   4, 395, 985    |        1, 458, 910, 247        |    17.59    |
| YOLO Nano<sub>leaky</sub> |   KD GAN    |0.395 ± 0.012|   2, 890, 527    |        2, 098, 305, 681        |    11.38    |
|YOLO Mobile<sub>leaky</sub>|   KD GAN    |0.311 ± 0.006|   4, 395, 985    |        1, 458, 910, 247        |    17.59    |

## ExDark test set


# References
* [YOLOv3](https://arxiv.org/abs/1804.02767)
* [LTH](https://arxiv.org/abs/1903.01611)
* [CS](https://arxiv.org/abs/1912.04427)
* [YOLO Nano](https://arxiv.org/abs/1910.01271)
* [Classical KD](https://papers.nips.cc/paper/2017/file/e1e32e235eee1f970470a3a6658dfdd5-Paper.pdf)
* [KD GAN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9046859)
* [MobileNet V3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)
