import os
import torch
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.pruning import create_mask_LTH, apply_mask_LTH
from models import Darknet, YOLO_Nano, SoftDarknet, FullSparseYOLO

x = torch.Tensor(1, 3, 416, 416)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/pascal/yolov3.cfg', help='args file to create the model.')
parser.add_argument('--output', type=str, default='', help='path to save the tensorboardX graph')
parser.add_argument('--model', type=str, help='Path to load the model.')
parser.add_argument('--mask', type=str, default=None, help='Path to load the mask, if existis.')
parser.add_argument('--embbed', action='store_true', help='To load the mask from the same checkpoint of model.')

args = vars(parser.parse_args())

# Initialize model
if 'soft' in args['model'] or 'soft' in args['cfg']:
    model = SoftDarknet(args['cfg'])
else:
    if 'nano' in args['cfg']: model = YOLO_Nano(args['cfg'])
    else: model = Darknet(args['cfg'])

if args['model']:
    checkpoint = torch.load(args['model'], map_location='cpu')
    try:
        try:
            model.load_state_dict(checkpoint['model'])
        except:        
            model.load_state_dict(checkpoint) 
    except:
        print("model key don't found in checkpoint. Trying without model key")
        model.load_state_dict(checkpoint)

# Applying mask
if (args['mask'] or args['embbed']):
    mask = create_mask_LTH(model)
    if args['mask']: mask.load_state_dict(torch.load(args['mask'], map_location='cpu'))
    else: mask.load_state_dict(checkpoint['mask'])
    apply_mask_LTH(model, mask)
elif 'soft' in args['cfg']:
    model.ticket = True
    model.temp = 1.
    _ = model(x)

if args['output']:
    if not os.path.exists(args['output']): os.makedirs(args['output'])

tb = SummaryWriter(args['output'])
tb.add_graph(model, x)
tb.close()