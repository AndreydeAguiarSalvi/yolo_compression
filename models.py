import torch.nn as nn
from utils.layers import *
import torch.nn.functional as F
from utils.google_utils import *
from utils.parse_config import *
from copy import deepcopy
from utils.utils import *
from collections import OrderedDict
ONNX_EXPORT = False


def create_modules(module_defs, img_size, arc):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()
        # if i == 0:
        #     modules.add_module('BatchNorm2d_0', nn.BatchNorm2d(output_filters[-1], momentum=0.1))

        if mdef['type'] in ['convolutional', 'multibias', 'multiconv_multibias', 'halfconv', 'inception', 'softconv']:
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            size = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if mdef['type'] == 'convolutional':
                modules.add_module('Conv2d', nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters, kernel_size=size, stride=stride,
                    padding=(size - 1) // 2 if mdef['pad'] else 0,
                    groups=mdef['groups'] if 'groups' in mdef else 1,
                    bias=not bn)
                )
            elif mdef['type'] == 'multibias':
                n_bias = mdef['n_bias']
                modules.add_module('Conv2d', MultiBiasConv(
                        in_channels = output_filters[-1], out_channels = filters, 
                        n_bias = n_bias,
                        kernel_size = size, stride = stride, pad = (size - 1) // 2 if mdef['pad'] else 0
                    )
                )
            elif mdef['type'] == 'multiconv_multibias':
                n_bias = mdef['n_bias']
                modules.add_module('Conv2d', MultiConvMultiBias(
                        in_channels = output_filters[-1], out_channels = filters, 
                        n_bias = n_bias,
                        kernel_size = size, stride = stride, pad = (size - 1) // 2 if mdef['pad'] else 0
                    )
                )
            elif mdef['type'] == 'halfconv':
                modules.add_module('Conv2d', HalfConv(
                        in_channels=output_filters[-1], out_channels=filters,
                        kernel_size=size, stride=stride, pad=(size - 1) // 2 if mdef['pad'] else 0,
                        have_bias=not bn
                    )
                )
            elif mdef['type'] == 'inception':
                modules.add_module('Conv2d', MyInception(
                        n_in=output_filters[-1],
                        n_out=filters
                    )
                )
            elif mdef['type'] == 'softconv':
                modules.add_module('Conv2d', SoftMaskedConv2d(
                    in_channels=output_filters[-1], out_channels=filters,
                    kernel_size=size, padding=(size-1) // 2 if mdef['pad'] else 0,
                    stride=stride, mask_initial_value=float(hyperparams['mask_initial_value'])
                    )
                )
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
            elif mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU(inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())

        elif mdef['type'] == 'PEP':
            filters = mdef['filters']
            x = mdef['x']
            stride = mdef['stride']
            act = mdef['activation']
            modules.add_module('PEP', PEP(
                    input_channels=output_filters[-1],
                    output_channels=filters,
                    x=x, stride=stride, activation=act
                )
            )
        
        elif mdef['type'] == 'EP':
            filters = mdef['filters']
            stride = mdef['stride']
            act = mdef['activation']
            modules.add_module('EP', EP(
                    input_channels=output_filters[-1],
                    output_channels=filters,
                    stride=stride, activation=act
                )
            )

        elif mdef['type'] == 'FCA':
            red = mdef['reduction']
            act = mdef['activation']
            modules.add_module('FCA', FCA(
                    channels=output_filters[-1],
                    reduction_ratio=red,
                    activation=act
                )
            )

        elif mdef['type'] == 'mobile':
            filters = mdef['filters']
            size = mdef['size']
            stride = mdef['stride']
            hidden = make_divisible(output_filters[-1] * mdef['expansion_ratio'], 8)
            act = mdef['activation']
            squeeze_and_excite = mdef['squeeze_excite']
            modules.add_module('MobileBottleneck', MobileBottleneck(
                    in_channels=output_filters[-1], hidden_dim=hidden,
                    out_channels=filters, kernel_size=size,
                    stride=stride, use_se=squeeze_and_excite, 
                    activation=act
                )
            )

        elif mdef['type'] == 'maxpool':
            size = mdef['size']
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=(size - 1) // 2)
            if size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + i for l in layers])
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = weightedFeatureFusion(layers=layers)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = mdef['mask']  # anchor mask
            modules = YOLOLayer(anchors=mdef['anchors'][mask],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1 or 2
                                arc=arc)  # yolo architecture

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                p = math.log(1 / (modules.nc - 0.99))  # class probability  ->  sigmoid(p) = 1/nc
                if arc == 'default' or arc == 'Fdefault':  # default
                    b = [-4.5, p]  # obj, cls
                elif arc == 'uBCE':  # unified BCE (80 classes)
                    b = [0, -9.0]
                elif arc == 'uCE':  # unified CE (1 background + 80 classes)
                    b = [10, -0.1]
                elif arc == 'uFBCE':  # unified FocalBCE (5120 obj, 80 classes)
                    b = [0, -6.5]
                elif arc == 'uFCE':  # unified FocalCE (64 cls, 1 background + 80 classes)
                    b = [7.7, -1.1]

                bias = module_list[-1][0].bias.view(len(mask), -1)  # 255 to 3x85
                bias[:, 4] += b[0] - bias[:, 4].mean()  # obj
                bias[:, 5:] += b[1] - bias[:, 5:].mean()  # cls
                # bias = torch.load('weights/yolov3-spp.bias.pt')[yolo_index]  # list of tensors [3x85, 3x85, 3x85]
                module_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))
                # utils.print_model_biases(model)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, arc):
        super(YOLOLayer, self).__init__()

        self.yolo_index = yolo_index
        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.arc = arc

        if ONNX_EXPORT:
            stride = [32, 16, 8][yolo_index]  # stride of this layer
            nx = img_size[1] // stride  # number x grid points
            ny = img_size[0] // stride  # number y grid points
            create_grids(self, img_size, (nx, ny))

    def forward(self, p, img_size):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Constants CAN NOT BE BROADCAST, ensure correct shape!
            m = self.na * self.nx * self.ny
            ng = 1 / self.ng.repeat((m, 1))
            grid_xy = self.grid_xy.repeat((1, self.na, 1, 1, 1)).view(m, 2)
            anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid_xy  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride

            if 'default' in self.arc:  # seperate obj and cls
                torch.sigmoid_(io[..., 4:])
            elif 'BCE' in self.arc:  # unified BCE (80 classes)
                torch.sigmoid_(io[..., 5:])
                io[..., 4] = 1
            elif 'CE' in self.arc:  # unified CE (1 background + 80 classes)
                io[..., 4:] = F.softmax(io[..., 4:], dim=4)
                io[..., 4] = 1

            if self.nc == 1:
                io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, self.no), p


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, arc)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, x, fts_indexes=[], verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out, fts = [], [], []
        verbose = False
        if verbose:
            str = ''
            print('0', x.shape)

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in [
                    'convolutional', 'multibias', 'multiconv_multibias', 
                    'halfconv', 'inception', 'upsample', 'maxpool',
                    'PEP', 'EP', 'FCA', 'mobile'
                ]:
                x = module(x)
            elif mtype == 'shortcut':  # sum
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    s = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, s)])
                x = module(x, out)  # weightedFeatureFusion()
            elif mtype == 'route': # concat
                layers = mdef['layers']
                if verbose:
                    l = [i - 1] + layers  # layers
                    s = [list(x.shape)] + [list(out[i].shape) for i in layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, s)])
                if len(layers) == 1:
                    x = out[layers[0]]
                else:
                    try:
                        x = torch.cat([out[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        out[layers[1]] = F.interpolate(out[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([out[i] for i in layers], 1)
                    # print(''), [print(out[i].shape) for i in layers], print(x.shape)
            elif mtype == 'yolo':
                yolo_out.append(module(x, img_size))
            out.append(x if i in self.routs else [])

            if i in fts_indexes: fts.append(x)

            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), mtype), list(x.shape), str)
                str = ''

        if self.training: # train
            if len(fts_indexes): return yolo_out, fts
            return yolo_out
        elif ONNX_EXPORT: # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            if len(fts_indexes): return x[0], torch.cat(x[1:3], 1), fts
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else: # test
            io, p = zip(*yolo_out)  # inference output, training output
            if len(fts_indexes): return torch.cat(io, 1), p, fts
            return torch.cat(io, 1), p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if weights and not os.path.isfile(weights):
        d = {'yolov3-spp.weights': '16lYS4bcIdM2HdmyJBVDOvt3Trx6N3W2R',
             'yolov3.weights': '1uTlyDWlnaqXcsKOktP5aH_zRDbfcDp-y',
             'yolov3-tiny.weights': '1CCF-iNIIkYesIDzaPvdwlcf7H9zSsKZQ',
             'yolov3-spp.pt': '1f6Ovy3BSq2wYq4UfvFUpxJFNDFfrIDcR',
             'yolov3.pt': '1SHNFyoe5Ni8DajDNEqgB2oVKBb_NoEad',
             'yolov3-tiny.pt': '10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo',
             'darknet53.conv.74': '1WUVBid-XuoUBmvzBVUCBl_ELrzqwA8dJ',
             'yolov3-tiny.conv.15': '1Bw0kCpplxUqyRYAJr9RY9SGnOJbo9nEj',
             'ultralytics49.pt': '158g62Vs14E3aj7oPVPuEnNZMKFNgGyNq',
            'ultralytics68.pt': '1Jm8kqnMdMGUUxGo8zMFZMJ0eaPwLkxSG',
             'yolov3-spp-ultralytics.pt': '1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4'}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)


class SparseYOLO(nn.Module):
    # YOLOv3 with sparse convolutions

    def __init__(self, pruned_yolo):
        super(SparseYOLO, self).__init__()

        self.module_defs = pruned_yolo.module_defs
        self.create_module_list(pruned_yolo)
        self.yolo_layers = pruned_yolo.yolo_layers

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.verbose = False

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)
    
    def create_module_list(self, pruned_yolo):
        self.module_list = nn.ModuleList()

        for module in pruned_yolo.module_list:
            my_module = deepcopy(module)
            if type(my_module) is nn.Sequential:
                my_module[0] = SparseConv(my_module[0])
            self.module_list.append(my_module)
        
        self.routs = pruned_yolo.routs


class MaskedNet(nn.Module):
    def __init__(self):
        super(MaskedNet, self).__init__()
        self.ticket = False

    def checkpoint(self):
        for m in self.mask_modules: m.checkpoint()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.checkpoint = deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules: m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                try:
                    m.load_state_dict(m.checkpoint)
                except:
                    print(f"cannot rewind weight from {'Conv2d' if isinstance(m, nn.Conv2d) else 'BatchNorm2d' if isinstance(m, nn.BatchNorm2d) else 'Linear'}")
                
    def prune(self):
        for m in self.mask_modules: m.prune(self.temp)


class SoftDarknet(MaskedNet):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        super(SoftDarknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, arc)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

        self.mask_modules = [m for m in self.modules() if type(m) == SoftMaskedConv2d]
        self.temp = 1

    def forward(self, x, verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        verbose = False
        if verbose:
            str = ''
            print('0', x.shape)

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'softconv', 'upsample', 'maxpool']:
                if mtype == 'softconv': 
                    x1 = module[0](x, self.temp, self.ticket)
                    x = module[1:](x1)
                else: x = module(x)
            elif mtype == 'shortcut':  # sum
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    s = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, s)])
                x = module(x, out)  # weightedFeatureFusion()
            elif mtype == 'route': # concat
                layers = mdef['layers']
                if verbose:
                    l = [i - 1] + layers  # layers
                    s = [list(x.shape)] + [list(out[i].shape) for i in layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, s)])
                if len(layers) == 1:
                    x = out[layers[0]]
                else:
                    try:
                        x = torch.cat([out[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        out[layers[1]] = F.interpolate(out[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([out[i] for i in layers], 1)
                    # print(''), [print(out[i].shape) for i in layers], print(x.shape)
            elif mtype == 'yolo':
                yolo_out.append(module(x, img_size))
            out.append(x if i in self.routs else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), mtype), list(x.shape), str)
                str = ''

        if self.training: # train
            return yolo_out
        elif ONNX_EXPORT: # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else: # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers


class HintModel(nn.Module):

    def __init__(self, config, teacher, student):
        super(HintModel, self).__init__()   
        
        self.hint_layers = nn.ModuleList()
        device = next(teacher.parameters()).device
        x = torch.Tensor(1, 3, 416, 416).to(device)
        
        # Get Features sizes
        with torch.no_grad(): 
            _, fts_tch = teacher(x, config['teacher_indexes'])
            _, fts_std = student(x, config['student_indexes'])

        for i, (ft_tch, ft_std) in enumerate(zip(fts_tch, fts_std)):
            _, chnl_tch, w_tch, h_tch = ft_tch.shape
            _, chnl_std, w_std, h_std = ft_std.shape
            if w_tch != w_std or h_tch != h_std: 
                print(f'Skiping {i}-th hint layer because shapes do not match:\n\tTeacher -> {[w_tch, h_tch]}\tStudent -> {[w_std, h_std]}')
            else: 
                print(f'\tCreating Hint for [{chnl_tch}, {w_tch}, {h_tch}] volume to [{chnl_std}, {w_std}, {h_std}]')
                hint_layer = nn.Sequential()
                conv = nn.Conv2d(
                        in_channels=chnl_std, out_channels=chnl_tch,
                        kernel_size=(1, 1), stride=1,
                        padding=0 # Missing padding trick from https://arxiv.org/pdf/1507.00448.pdf to match the resolution
                )
                hint_layer.add_module(f'hint_{i}', conv)
                hint_layer.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                
                self.hint_layers.append(hint_layer)
    

    def forward(self, fts):
        y = []
        for i, (x, module) in enumerate(zip(fts, self.hint_layers)):
            y.append(module(x))
        return y

    def to(self, device):
        for module in self.hint_layers: module = module.to(device)
        

class Discriminator(nn.Module):

    def __init__(self, fts_indexes, model, kernel=(1, 1), has_sigmoid=True):
        super(Discriminator, self).__init__()
        
        self.D = nn.ModuleList()
        device = next(model.parameters()).device
        x = torch.Tensor(1, 3, 416, 416).to(device)
        
        # Get Features sizes
        with torch.no_grad(): 
            _, fts = model(x, fts_indexes)
        
        for i, ft in enumerate(fts):
            _, chnl, w, h = ft.shape
            in_sizes = [chnl, 256, 512, 256, 128]
            out_sizes = [256, 512, 256, 128, 64]
            dct = nn.Sequential()
            print(f'\tCreating Discriminator for [{chnl}, {w}, {h}] volume')
            for j in range(len(in_sizes)): 
                cnv = nn.Conv2d(
                    in_channels = in_sizes[j],
                    out_channels = out_sizes[j],
                    kernel_size = kernel,
                    padding = (0, 0) if kernel[0] == 1 else (1, 1)
                )
                dct.add_module(f'D-{i}__Layer-{j}', cnv)
                dct.add_module(f'D-{i}__Norm-{j}', nn.BatchNorm2d(out_sizes[j], momentum=0.1))
                dct.add_module(f'activation-{j}', nn.LeakyReLU(0.1, inplace=True))    
            
            dct.add_module(f'D-{i}__AvgPool2d', nn.AdaptiveAvgPool2d((7, 7)))
            dct.add_module(f'Reshape', View())

            dct.add_module(f'D-{i}__Linear-1', nn.Linear(in_features=64*7*7, out_features=1024))
            dct.add_module(f'activation-{j+1}', nn.LeakyReLU(0.1, inplace=True))

            dct.add_module(f'D-{i}__Linear-2', nn.Linear(in_features=1024, out_features=128))
            dct.add_module(f'activation-{j+2}', nn.LeakyReLU(0.1, inplace=True))

            dct.add_module(f'D-{i}__Linear-3', nn.Linear(in_features=128, out_features=1))
            if has_sigmoid: dct.add_module(f'activation-{j+3}', nn.Sigmoid())

            self.D.append(dct)
    
    def forward(self, fts):
        y = []
        for i, (x, discriminator) in enumerate(zip(fts, self.D)):
            y.append(discriminator(x))
        return y
