import torch
import argparse
from models import *
from thop import profile
from utils.layers import M2MSparseConv, SoftMaskedConv2d
from utils.pruning import create_mask_LTH, apply_mask_LTH


def count_M2MSparsable(m, x, y):
    m.total_ops = torch.DoubleTensor([m.parameters().numel() * m.IN_SH[-1]])

def count_SoftConv(m, x, y):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()
    total_ops = y.nelement() * (m.in_channels * kernel_ops)
    m.total_ops += torch.DoubleTensor([int(total_ops)])

custom_hooks = {
    M2MSparseConv: count_M2MSparsable,
    SoftMaskedConv2d: count_SoftConv
}


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to load the model.')
parser.add_argument('--cfg', type=str, help='args file to create the model.')
parser.add_argument('--mask', type=str, default=None, help='Path to load the mask, if existis.')
parser.add_argument('--embbed', action='store_true', help='To load the mask from the same checkpoint of model.')
parser.add_argument('--device', help='cuda:id or cpu', required=True)
parser.add_argument('--macs_reduction', default='full_sparse', choices=['full_sparse', 'channel_sparse', 'none'], 
    help='Which approach for MACs reduction. none means use the default model')
args = vars(parser.parse_args())

device = torch.device(args['device'])
x = torch.Tensor(1, 3, 416, 416).to(device)

# Initialize model
if 'soft' in args['cfg']: model = SoftDarknet(args['cfg']).to(device)
elif 'nano' in args['cfg']: model = YOLO_Nano(args['cfg']).to(device)
else: model = Darknet(args['cfg']).to(device)

if args['model']:
    checkpoint = torch.load(args['model'], map_location=device)
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        model.load_state_dict(checkpoint)

if (args['mask'] or args['embbed']):
    mask = create_mask_LTH(model)
    if args['mask']: mask.load_state_dict(torch.load(args['mask'], map_location=device))
    else: mask.load_state_dict(checkpoint['mask'])
    apply_mask_LTH(model, mask)

elif 'soft' in args['cfg']:
    model.ticket = True
    model.temp = 1.
    _ = model(x)

if not (args['mask'] or args['embbed'] or 'soft' in args['model']) or args['macs_reduction'] == 'none': 
    print('using model with no MACs reduction')
    total_ops, total_params = profile(model, (x,), custom_ops=custom_hooks, verbose=True)

elif args['macs_reduction'] == 'full_sparse':
    print('using FullSparseYOLO for MACs reduction')
    sparse = FullSparseYOLO(model).to(device)
    total_ops, total_params = profile(sparse, (x, ), custom_ops=custom_hooks, verbose=True)

elif args['macs_reduction'] == 'channel_sparse':
    print('using Ch_Wise_SparseYOLO for MACs reduction')
    sparse = Ch_Wise_SparseYOLO(model).to(device)
    total_ops, total_params = profile(sparse, (x, ), custom_ops=custom_hooks, verbose=True)

print("%s | %s" % ("Params", "MACs"))
print("---|---")
print(f"{total_params:,}\t{total_ops:,}")