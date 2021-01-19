import os
import torch
import argparse
import torch.nn as nn
from models import Darknet, YOLO_Nano
from torch.utils.tensorboard import SummaryWriter

x = torch.Tensor(1, 3, 416, 416)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/pascal/yolov3.cfg', help='args file to create the model.')
parser.add_argument('--output', type=str, default='', help='path to save the tensorboardX graph')
args = vars(parser.parse_args())

model = Darknet(args['cfg'])
y = model(x)

if args['output']:
    if not os.path.exists(args['output']): os.makedirs(args['output'])

tb = SummaryWriter(args['output'])
tb.add_graph(model, x)
tb.close()