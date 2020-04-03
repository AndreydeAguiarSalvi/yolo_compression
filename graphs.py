import torch
import torch.nn as nn
from models import Darknet, YOLO_Nano
from torch.utils.tensorboard import SummaryWriter

x = torch.Tensor(1, 3, 416, 416)

model = YOLO_Nano()
y = model(x)

tb = SummaryWriter('debugs/yolo_nano')
tb.add_graph(model, x)
tb.close()