import torch
from models import Darknet

net = Darknet(cfg='cfg/yolov3.cfg')
x = torch.Tensor(1, 3, 416, 416)

fts = net.module_list[0](x)