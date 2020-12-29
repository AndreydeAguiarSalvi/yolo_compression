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

# References
* [YOLOv3](https://arxiv.org/abs/1804.02767)
* [LTH](https://arxiv.org/abs/1903.01611)
* [CS](https://arxiv.org/abs/1912.04427)
* [YOLO Nano](https://arxiv.org/abs/1910.01271)
* [Classical KD](https://papers.nips.cc/paper/2017/file/e1e32e235eee1f970470a3a6658dfdd5-Paper.pdf)
* [KD GAN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9046859)
* [MobileNet V3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)
