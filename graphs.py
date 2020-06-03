import torch
import torch.nn as nn
from models import Darknet, YOLO_Nano
from torch.utils.tensorboard import SummaryWriter

x = torch.Tensor(1, 3, 416, 416)

model = Darknet(cfg='cfg/voc_yolov3.cfg')
y = model(x)

tb = SummaryWriter('debugs/yolov3')
tb.add_graph(model, x)
tb.close()