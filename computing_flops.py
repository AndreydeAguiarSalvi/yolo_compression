import torch
import argparse
# from thop import profile, clever_format
from models import Darknet, SparseYOLO, SoftDarknet, YOLO_Nano

x = torch.Tensor(1, 3, 416, 416)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to load the model.', required=True)
parser.add_argument('--darknet', type=str, help='Architecture to create.', required=True)
parser.add_argument('--cfg', type=str, help='args file to create the model.')
parser.add_argument('--mask', type=str, help='Path to load the mask, if existis.')
parser.add_argument('--embbed', action='store_true', help='To load the mask from the same checkpoint of model.')
parser.add_argument('device', help='cuda:id or cpu', required=True)
# parser.add_argument('--clever_format', action='store_true')
args = vars(parser.parse_args())

device = torch.device(args['device'])

# Initialize model
if args['darknet'] == 'default':
    model = Darknet(args['cfg'], arc=args['arc']).to(device)
elif args['darknet'] == 'nano':
    model = YOLO_Nano().to(device)
elif args['darknet'] == 'soft':
    model = SoftDarknet(args['cfg'], arc=args['arc']).to(device)
    model.ticket = True
    _ = model(x)

sparse = SparseYOLO(model).to(device)

# if args['clever_format']: macs, params = ?