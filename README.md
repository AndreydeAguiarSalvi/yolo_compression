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
- `numpy = 1.17 (version 1.18 raises bugs on COCOAPI)`
- `torch >= 1.3`
- `opencv-python`
- `Pillow`

Even more, install [THOP](https://github.com/Lyken17/pytorch-OpCounter) to count the MACs
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

# Other Details
I am now focused on completing my master's (scheduled for March, 2020). With this task completed, I will bring you the final results of the work and examples of how to run this repository.
Basically, run 
* train.py to perform a normal training,
* prune.py to perform pruning with LTH or CS, depending on the params
* my_kd.py to perform classical KD with YOLOv3 and YOLO Mobile (model of my own) or YOLO Nano
* my_kd_gan.py to perform my adapted GAN based KD
In utils/my_utils.py, you can see the argument parser, to see all the available parameters

# References
% YOLOv3
@misc{redmon2018yolov3,
    author = "Redmon, Joseph and Farhadi, Ali",
    title = "YOLOv3: An Incremental Improvement",
    year = "2018",
    eprint = "1804.02767",
    archivePrefix = "arXiv",
    primaryClass = "cs.CV",
    url = "https://arxiv.org/abs/1804.02767",
    urlaccessdate = "05/11/2020"
}

% LTH with Latte Reseting

@misc{frankle2019stabilizing,
    author = "Frankle, Jonathan and Dziugaite, Gintare Karolina and Roy, Daniel M. and Carbin, Michael",
    title = "Stabilizing the Lottery Ticket Hypothesis",
    year = "2019",
    eprint = "1903.01611",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    url = "https://arxiv.org/abs/1903.01611",
    urlaccessdate = "07/24/2020"
}

% CS
@misc{savarese2019winning,
    author = "Savarese, Pedro and Silva, Hugo and Maire, Michael",
    title = "Winning the Lottery with Continuous Sparsification",
    year = "2019",
    eprint = "1912.04427",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    url = "https://arxiv.org/abs/1912.04427",
    urlaccessdate = "06/14/2020"
}

% YOLO Nano
@misc{alex2019yolo,
    author = "Wong, Alexander and Famuori, Mahmoud and Shafiee, Mohammad Javad and Li, Francis and Chwyl, Brendan and Chung, Jonathan",
    title = "YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection",
    year = "2019",
    eprint = "1910.01271",
    archivePrefix = "arXiv",
    primaryClass = "cs.CV",
    url = "https://arxiv.org/abs/1910.01271",
    urlaccessdate = "07/20/2020"
}

% KD on Faster
@inproceedings{guobin_2017,
    author = "Chen, Guobin and Choi, Wongun and Yu, Xiang and Han, Tony and Chandraker, Manmohan",
    title = "Learning Efficient Object Detection Models with Knowledge Distillation",
    year = "2017",
    isbn = "9781510860964",
    publisher = "Curran Associates Inc.",
    address = "Red Hook, NY, USA",
    booktitle = "Proceedings of the 31st International Conference on Neural Information Processing Systems",
    pages = "742--751",
    numpages = "10",
    location = "Long Beach, California, USA",
    series = "NIPS’17"
}

% KD with GAN
@ARTICLE{wang_2020,
    author = "{Wang}, W. and {Hong}, W. and {Wang}, F. and {Yu}, J.",
    journal = "IEEE Access",
    title = "GAN-Knowledge Distillation for One-Stage Object Detection",
    year = "2020",
    volume = "8",
    number = "",
    pages = "60719-60727",
    month = "Mar"
}

% MobileNet V3
@InProceedings{Howard_2019_ICCV,
    author = {Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V. and Adam, Hartwig},
    title = {Searching for MobileNetV3},
    booktitle = {International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2019},
    pages = {1314--1324}
}