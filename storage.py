import os
import torch
import argparse
from models import *
from collections import OrderedDict 
from utils.pruning import create_mask_LTH, apply_mask_LTH


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def fullsparse_state_dict(model):
    # Getting the original state_dict
    original_state = model.state_dict()
    state = {}
    for key in original_state: state[key] = original_state[key]

    for i, m in enumerate(model.module_list):
        if isinstance(m, nn.Sequential):
            if len(m):
                if isinstance(m[0], M2MSparseConv):
                    key = f"module_list.{i}.M2MSparseConv.W.values"
                    state[key] = m[0].W._values()
                    key = f"module_list.{i}.M2MSparseConv.W.indices"
                    state[key] = m[0].W._indices()

                    if m[0].B is not None:
                        key = f"module_list.{i}.M2MSparseConv.B"
                        state[key] = m[0].B

    return OrderedDict(sorted(state.items(), key=lambda t: t[0]))


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/pascal/yolov3.cfg', help='args file to create the model.')
parser.add_argument('--weights', type=str, help='Path to load the model.')
parser.add_argument('--mask', type=str, default=None, help='Path to load the mask, if existis.')
parser.add_argument('--embbed', action='store_true', help='To load the mask from the same checkpoint of model.')
args = vars(parser.parse_args())

checkpoint = torch.load(args['weights'], map_location='cpu')
md_ck = checkpoint['model'] if 'model' in checkpoint else checkpoint

if 'soft' in args['cfg']:
    model = SoftDarknet(args['cfg'])
    model.load_state_dict(md_ck)
    model.ticket = True
    model.temp = 1.
    _ = model(torch.Tensor(1, 3, 416, 416))
    model = FullSparseYOLO(model)
    md_ck = fullsparse_state_dict(model)


elif args['mask'] or args['embbed']:
    model = Darknet(args['cfg'])
    model.load_state_dict(md_ck)
    mask = create_mask_LTH(model)
    if args['mask']: mask.load_state_dict(torch.load(args['mask'], map_location='cpu'))
    else: mask.load_state_dict(checkpoint['mask'])
    apply_mask_LTH(model, mask)
    model = FullSparseYOLO(model)
    md_ck = fullsparse_state_dict(model)

torch.save(md_ck, 'current.pt')
size = os.path.getsize('current.pt')
print(convert_size(size))