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
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                    out_channels=filters,
                                                    kernel_size=size,
                                                    stride=stride,
                                                    padding=(size - 1) // 2 if mdef['pad'] else 0,
                                                    groups=mdef['groups'] if 'groups' in mdef else 1,
                                                    bias=not bn))
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


class weightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(weightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1 # number of layers
        if weight:
            self.w = torch.nn.Parameter(torch.zeros(self.n))  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nc = x.shape[1]  # input of channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            ac = a.shape[1] # feature channels
            dc = nc - ac # delta channels

            # Adjust channels
            if dc > 0:  # slice input
                x[:, :ac] = x[:, :ac] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            elif dc < 0:  # slice feature
                x = x + a[:, :nc]
            else:  # same shape
                x = x + a
        return x

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, arc):
        super(YOLOLayer, self).__init__()

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

    def forward(self, x, verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        verbose = False
        if verbose:
            str = ''
            print('0', x.shape)

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'multibias', 'multiconv_multibias', 'halfconv', 'inception', 'upsample', 'maxpool']:
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

################
# My Additions #
################
class MultiBiasConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_bias, kernel_size=(3, 3), stride=1, pad=0):
        super(MultiBiasConv, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=int(in_channels), out_channels=int(out_channels/n_bias), bias=False, kernel_size=kernel_size, padding=pad, stride=stride)
        nn.init.xavier_normal_(self.conv.weight)
        self.bias = torch.nn.Parameter( torch.Tensor(n_bias), requires_grad=True )
        nn.init.normal_(self.bias)
    
    def forward(self, x):
        x = self.conv(x)
        y = []
        for b in self.bias:
            y.append(x + b)
        
        return torch.cat(y, dim=1)


class MultiConvMultiBias(nn.Module):
    def __init__(self, in_channels, out_channels, n_bias, kernel_size=3, stride=1, pad=0):
        super(MultiConvMultiBias, self).__init__()
        # Input is not divisible by 2^n
        bias_1, bias_2 = n_bias, n_bias
        if in_channels == 3:
            bias_1 = 1
        self.conv1 = MultiBiasConv(in_channels=in_channels, out_channels=in_channels, n_bias=bias_1, 
            kernel_size=(kernel_size, 1), pad=pad, stride=stride
        )
        self.conv2 = MultiBiasConv(in_channels=in_channels, out_channels=out_channels, n_bias=bias_2, 
            kernel_size=(1, kernel_size), pad=pad, stride=stride
        )
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        return x


class HalfConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad, have_bias):
        super(HalfConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=int(out_channels/2),
            kernel_size=kernel_size, stride = stride, padding=pad, bias=have_bias
        )
        

    def forward(self, x):
        x = self.conv(x)
        x = torch.cat((x, -x), 1)

        return x


class MyInception(nn.Module):

    def __init__(self, n_in, n_out):
        super(MyInception, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels=n_in, out_channels=int(n_out/2), kernel_size=1, stride=1, padding=0)
        self.conv1x1_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=int(n_out/4), kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=int(n_out/4), out_channels=int(n_out/4), kernel_size=3, stride=1, padding=1)
        )
        self.conv1x1_3x3_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=int(n_out/4), kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=int(n_out/4), out_channels=int(n_out/4), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=int(n_out/4), out_channels=int(n_out/4), kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv1x1_3x3(x)
        x3 = self.conv1x1_3x3_3x3(x)

        return torch.cat((x1, x2, x3), 1)


def sigmoid(x):
    return float(1./(1.+np.exp(-x)))


class SoftMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, mask_initial_value=0.):
        super(SoftMaskedConv2d, self).__init__()
        self.mask_initial_value = mask_initial_value
        
        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        self.init_weight = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.init_mask()
        
    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)

    def compute_mask(self, temp, ticket):
        scaling = 1. / sigmoid(self.mask_initial_value)
        if ticket: mask = (self.mask_weight > 0).float()
        else: mask = F.sigmoid(temp * self.mask_weight)
        return scaling * mask      
        
    def prune(self, temp):
        self.mask_weight.data = torch.clamp(temp * self.mask_weight.data, max=self.mask_initial_value)   

    def forward(self, x, temp=1, ticket=False):
        self.mask = self.compute_mask(temp, ticket)
        masked_weight = self.weight * self.mask
        out = F.conv2d(x, masked_weight, stride=self.stride, padding=self.padding)        
        return out
        
    def checkpoint(self):
        self.init_weight.data = self.weight.clone()       
        
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def extra_repr(self):
        return '{}, {}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)


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


# Adapted from https://github.com/liux0614/yolo_nano/blob/master/models/yolo_nano.py
def conv1x1(input_channels, output_channels, stride=1, bn=True):
    # 1x1 convolution without padding
    if bn == True:
        return nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=1,
                stride=stride, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )
    else:
        return nn.Conv2d(
                input_channels, output_channels, kernel_size=1,
                stride=stride, bias=False)


def conv3x3(input_channels, output_channels, stride=1, bn=True):
    # 3x3 convolution with padding=1
    if bn == True:
        return nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=3,
                stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )
    else:
        nn.Conv2d(
                input_channels, output_channels, kernel_size=3,
                stride=stride, padding=1, bias=False)


def sepconv3x3(input_channels, output_channels, stride=1, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(
            input_channels, input_channels * expand_ratio,
            kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(input_channels * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(
            input_channels * expand_ratio, input_channels * expand_ratio, kernel_size=3, 
            stride=stride, padding=1, groups=input_channels * expand_ratio, bias=False),
        nn.BatchNorm2d(input_channels * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(
            input_channels * expand_ratio, output_channels,
            kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(output_channels)
    )


class EP(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(EP, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        self.sepconv = sepconv3x3(input_channels, output_channels, stride=stride)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.sepconv(x)
        
        return self.sepconv(x)


class PEP(nn.Module):
    def __init__(self, input_channels, output_channels, x, stride=1):
        super(PEP, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        self.conv = conv1x1(input_channels, x)
        self.sepconv = sepconv3x3(x, output_channels, stride=stride)
        
    def forward(self, x):        
        out = self.conv(x)
        out = self.sepconv(out)
        if self.use_res_connect:
            return out + x

        return out


class FCA(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super(FCA, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        hidden_channels = channels // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        out = x * out.expand_as(x)
        return out


class YOLO_Nano(nn.Module):
    def __init__(self, num_classes=20, image_size=416, anchor_type='PASCAL'):
        super(YOLO_Nano, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_anchors = 3
        self.yolo_channels = (self.num_classes + 5) * self.num_anchors
        
        if anchor_type == 'PASCAL':
            self.anchors = [ [26,31],  [43,84],  [81,171],   [103,68],  [145,267],  [180,135],  [247,325],  [362,178],  [412,346] ]
        elif anchor_type == 'COCO':
            self.anchors = [ [10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326] ]

        # image:  416x416x3
        self.conv1 = conv3x3(3, 12, stride=1) # output: 416x416x12
        self.conv2 = conv3x3(12, 24, stride=2) # output: 208x208x24
        self.pep1 = PEP(24, 24, 7, stride=1) # output: 208x208x24
        self.ep1 = EP(24, 70, stride=2) # output: 104x104x70
        self.pep2 = PEP(70, 70, 25, stride=1) # output: 104x104x70
        self.pep3 = PEP(70, 70, 24, stride=1) # output: 104x104x70
        self.ep2 = EP(70, 150, stride=2) # output: 52x52x150
        self.pep4 = PEP(150, 150, 56, stride=1) # output: 52x52x150
        self.conv3 = conv1x1(150, 150, stride=1) # output: 52x52x150
        self.fca1 = FCA(150, 8) # output: 52x52x150
        self.pep5 = PEP(150, 150, 73, stride=1) # output: 52x52x150
        self.pep6 = PEP(150, 150, 71, stride=1) # output: 52x52x150
        
        self.pep7 = PEP(150, 150, 75, stride=1) # output: 52x52x150
        self.ep3 = EP(150, 325, stride=2) # output: 26x26x325
        self.pep8 = PEP(325, 325, 132, stride=1) # output: 26x26x325
        self.pep9 = PEP(325, 325, 124, stride=1) # output: 26x26x325
        self.pep10 = PEP(325, 325, 141, stride=1) # output: 26x26x325
        self.pep11 = PEP(325, 325, 140, stride=1) # output: 26x26x325
        self.pep12 = PEP(325, 325, 137, stride=1) # output: 26x26x325
        self.pep13 = PEP(325, 325, 135, stride=1) # output: 26x26x325
        self.pep14 = PEP(325, 325, 133, stride=1) # output: 26x26x325
        
        self.pep15 = PEP(325, 325, 140, stride=1) # output: 26x26x325
        self.ep4 = EP(325, 545, stride=2) # output: 13x13x545
        self.pep16 = PEP(545, 545, 276, stride=1) # output: 13x13x545
        self.conv4 = conv1x1(545, 230, stride=1) # output: 13x13x230
        self.ep5 = EP(230, 489, stride=1) # output: 13x13x489
        self.pep17 = PEP(489, 469, 213, stride=1) # output: 13x13x469
        
        self.conv5 = conv1x1(469, 189, stride=1) # output: 13x13x189
        self.conv6 = conv1x1(189, 105, stride=1) # output: 13x13x105
        # upsampling conv6 to 26x26x105
        # concatenating [conv6, pep15] -> pep18 (26x26x430)
        self.pep18 = PEP(430, 325, 113, stride=1) # output: 26x26x325
        self.pep19 = PEP(325, 207, 99, stride=1) # output: 26x26x325
        
        self.conv7 = conv1x1(207, 98, stride=1) # output: 26x26x98
        self.conv8 = conv1x1(98, 47, stride=1) # output: 26x26x47
        # upsampling conv8 to 52x52x47
        # concatenating [conv8, pep7] -> pep20 (52x52x197)
        self.pep20 = PEP(197, 122, 58, stride=1) # output: 52x52x122
        self.pep21 = PEP(122, 87, 52, stride=1) # output: 52x52x87
        self.pep22 = PEP(87, 93, 47, stride=1) # output: 52x52x93
        self.conv9 = conv1x1(93, self.yolo_channels, stride=1, bn=False) # output: 52x52x yolo_channels
        self.yolo_layer52 = YOLOLayer(
            anchors=self.anchors[0:3], nc=self.num_classes,
            img_size=(image_size, image_size), yolo_index=0, arc='default'
        )

        # conv7 -> ep6
        self.ep6 = EP(98, 183, stride=1) # output: 26x26x183
        self.conv10 = conv1x1(183, self.yolo_channels, stride=1, bn=False) # output: 26x26x yolo_channels
        self.yolo_layer26 = YOLOLayer(
            anchors=self.anchors[3:6], nc=self.num_classes,
            img_size=(image_size, image_size), yolo_index=1, arc='default'
        )

        # conv5 -> ep7
        self.ep7 = EP(189, 462, stride=1) # output: 13x13x462
        self.conv11 = conv1x1(462, self.yolo_channels, stride=1, bn=False) # output: 13x13x yolo_channels
        self.yolo_layer13 = YOLOLayer(
            anchors=self.anchors[6:], nc=self.num_classes,
            img_size=(image_size, image_size), yolo_index=2, arc='default'
        )

        self.module_list = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()
            self.module_list.append(m)
    
        self.create_modules()
        self.yolo_layers = [37, 40, 43]

    def forward(self, x, fts_indexes=[]):
        yolo_outputs = []
        features = []
        current_size = x.shape[-2:]

        out = self.conv1(x)
        if 0 in fts_indexes: features.append(out)
        out = self.conv2(out)
        if 1 in fts_indexes: features.append(out)
        out = self.pep1(out)
        if 2 in fts_indexes: features.append(out)
        out = self.ep1(out)
        if 3 in fts_indexes: features.append(out)
        out = self.pep2(out)
        if 4 in fts_indexes: features.append(out)
        out = self.pep3(out)
        if 5 in fts_indexes: features.append(out)
        out = self.ep2(out)
        if 6 in fts_indexes: features.append(out)
        out = self.pep4(out)
        if 7 in fts_indexes: features.append(out)
        out = self.conv3(out)
        if 8 in fts_indexes: features.append(out)
        out = self.fca1(out)
        if 9 in fts_indexes: features.append(out)
        out = self.pep5(out)
        if 10 in fts_indexes: features.append(out)
        out = self.pep6(out)
        if 11 in fts_indexes: features.append(out)
        
        out_pep7 = self.pep7(out)
        if 12 in fts_indexes: features.append(out_pep7)
        out = self.ep3(out_pep7)
        if 13 in fts_indexes: features.append(out)
        out = self.pep8(out)
        if 14 in fts_indexes: features.append(out)
        out = self.pep9(out)
        if 15 in fts_indexes: features.append(out)
        out = self.pep10(out)
        if 16 in fts_indexes: features.append(out)
        out = self.pep11(out)
        if 17 in fts_indexes: features.append(out)
        out = self.pep12(out)
        if 18 in fts_indexes: features.append(out)
        out = self.pep13(out)
        if 19 in fts_indexes: features.append(out)
        out = self.pep14(out)
        if 20 in fts_indexes: features.append(out)

        out_pep15 = self.pep15(out)
        if 21 in fts_indexes: features.append(out_pep15)
        out = self.ep4(out_pep15)
        if 22 in fts_indexes: features.append(out)
        out = self.pep16(out)
        if 23 in fts_indexes: features.append(out)
        out = self.conv4(out)
        if 24 in fts_indexes: features.append(out)
        out = self.ep5(out)
        if 25 in fts_indexes: features.append(out)
        out = self.pep17(out)
        if 26 in fts_indexes: features.append(out)

        out_conv5 = self.conv5(out)
        if 27 in fts_indexes: features.append(out_conv5)
        out = F.interpolate(self.conv6(out_conv5), scale_factor=2)
        if 28 in fts_indexes: features.append(out)
        out = torch.cat([out, out_pep15], dim=1)
        if 29 in fts_indexes: features.append(out)
        out = self.pep18(out)
        if 30 in fts_indexes: features.append(out)
        out = self.pep19(out)
        if 31 in fts_indexes: features.append(out)
        
        out_conv7 = self.conv7(out)
        if 32 in fts_indexes: features.append(out_conv7)
        out = F.interpolate(self.conv8(out_conv7), scale_factor=2)
        if 33 in fts_indexes: features.append(out)
        out = torch.cat([out, out_pep7], dim=1)
        if 34 in fts_indexes: features.append(out)
        out = self.pep20(out)
        if 35 in fts_indexes: features.append(out)
        out = self.pep21(out)
        if 36 in fts_indexes: features.append(out)
        out = self.pep22(out)
        if 37 in fts_indexes: features.append(out)
        out_conv9 = self.conv9(out)
        if 38 in fts_indexes: features.append(out_conv9)
        yolo_outputs.append(self.yolo_layer52(out_conv9, current_size))

        out = self.ep6(out_conv7)
        if 40 in fts_indexes: features.append(out)
        out_conv10 = self.conv10(out)
        if 41 in fts_indexes: features.append(out_conv10)
        yolo_outputs.append(self.yolo_layer26(out_conv10, current_size))

        out = self.ep7(out_conv5)
        if 43 in fts_indexes: features.append(out)
        out_conv11 = self.conv11(out)
        if 44 in fts_indexes: features.append(out_conv11)
        yolo_outputs.append(self.yolo_layer13(out_conv11, current_size))

        if self.training: # train
            return yolo_outputs, features if len(fts_indexes) else yolo_outputs
        elif ONNX_EXPORT: # export
            x = [torch.cat(x, 0) for x in zip(*yolo_outputs)]
                                                                                # scores, boxes: 3780x80, 3780x4
            return x[0], torch.cat(x[1:3], 1), features if len(fts_indexes) else x[0], torch.cat(x[1:3], 1)
        else: # test
            io, p = zip(*yolo_outputs)  # inference output, training output
            return torch.cat(io, 1), p, features if len(fts_indexes) else torch.cat(io, 1), p
    

    def create_modules(self):
        self.module_list = []

        self.module_list.append(self.conv1)
        self.module_list.append(self.conv2)
        self.module_list.append(self.pep1)
        self.module_list.append(self.ep1)
        self.module_list.append(self.pep2)
        self.module_list.append(self.pep3)
        self.module_list.append(self.ep2)
        self.module_list.append(self.pep4)
        self.module_list.append(self.conv3)
        self.module_list.append(self.fca1)
        self.module_list.append(self.pep5)
        self.module_list.append(self.pep6)
        
        self.module_list.append(self.pep7)
        self.module_list.append(self.ep3)
        self.module_list.append(self.pep8)
        self.module_list.append(self.pep9)
        self.module_list.append(self.pep10)
        self.module_list.append(self.pep11)
        self.module_list.append(self.pep12)
        self.module_list.append(self.pep13)
        self.module_list.append(self.pep14)

        self.module_list.append(self.pep15)
        self.module_list.append(self.ep4)
        self.module_list.append(self.pep16)
        self.module_list.append(self.conv4)
        self.module_list.append(self.ep5)
        self.module_list.append(self.pep17)
        
        self.module_list.append(self.conv5)
        self.module_list.append(self.conv6)
        
        self.module_list.append(self.pep18)
        self.module_list.append(self.pep19)

        self.module_list.append(self.conv7)
        self.module_list.append(self.conv8)
        
        self.module_list.append(self.pep20)
        self.module_list.append(self.pep21)
        self.module_list.append(self.pep22)
        self.module_list.append(self.conv9)
        self.module_list.append(self.yolo_layer52)
        
        self.module_list.append(self.ep6)
        self.module_list.append(self.conv10)
        self.module_list.append(self.yolo_layer26)

        self.module_list.append(self.ep7)
        self.module_list.append(self.conv11)
        self.module_list.append(self.yolo_layer13)


class SparseConv(nn.Module):

    def __init__(self, original_conv):
        super(SparseConv, self).__init__()
        self.find_non_null_filters(original_conv)
        dict = self.create_splited_convs(original_conv)
        self.fractional_convs = nn.ModuleDict(dict)
    

    def forward(self, x):
        y = []
        # y2 = []
        for (key, value) in self.fractional_convs.items():
            y.append(value(x))
        # for i in reversed(y): y2.append(i)

        return torch.cat(y, dim=1)
        

    def find_non_null_filters(self, conv): # conv.shape is out_channels, in_channels, x, y
        device = next(conv.parameters()).device
        if type(conv) is SoftMaskedConv2d: params = conv.weight * conv.mask
        else: params = conv.weight
        onehot_parameters = torch.sum(torch.abs(params), dim=(1, 2, 3))
        self.convs_list = torch.where( onehot_parameters > 0, torch.tensor(1, device=device), torch.tensor(0, device=device) )
        

    def create_splited_convs(self, original_conv):
        sequential_ones = []
        sequential_zeros = []
        result = OrderedDict()
        count_ones = 1
        count_zeros = 1

        for i in range(self.convs_list.shape[0]):
            if self.convs_list[i] == torch.tensor(1): 
                if len(sequential_zeros ) > 0:
                    new_conv = ZeroConv(sequential_zeros, original_conv.kernel_size, original_conv.padding, original_conv.stride)
                    result['zero' + str(count_zeros)] = new_conv
                    count_zeros += 1
                    sequential_zeros = []

                sequential_ones.append(i)
            else:
                if len(sequential_ones) > 0:
                    new_conv = self.create_miniconv_from(original_conv, sequential_ones)
                    result['conv' + str(count_ones)] = new_conv
                    count_ones += 1
                    sequential_ones = []
                
                sequential_zeros.append(1)
        
        if len(sequential_ones) > 0:
            new_conv = self.create_miniconv_from(original_conv, sequential_ones)
            result['conv' + str(count_ones)] = new_conv
        elif len(sequential_zeros) > 0:
            new_conv = ZeroConv(sequential_zeros, original_conv.kernel_size, original_conv.padding, original_conv.stride)
            result['zero' + str(count_zeros)] = new_conv
        
        return result


    def create_miniconv_from(self, original_conv, channels_list):
        if type(original_conv) == SoftMaskedConv2d:
            new_conv = nn.Conv2d(
                in_channels=original_conv.in_channels, out_channels=len(channels_list), 
                kernel_size=original_conv.kernel_size, padding=original_conv.padding,
                stride=original_conv.stride,
                bias = False
            )
            data = original_conv.weight * original_conv.mask
            new_conv.weight.data = data[ channels_list[0] : channels_list[-1]+1 ]
        else:
            new_conv = nn.Conv2d(
                in_channels=original_conv.in_channels, out_channels=len(channels_list), 
                kernel_size=original_conv.kernel_size, padding=original_conv.padding,
                stride=original_conv.stride, groups=original_conv.groups,
                bias = True if original_conv.bias is not None else False
            )
            new_conv.weight.data = original_conv.weight[ channels_list[0] : channels_list[-1]+1 ]
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias[ channels_list[0] : channels_list[-1]+1 ]

        return new_conv


class ZeroConv(nn.Module):

    def __init__(self, channels_list, kernel_size, padding, stride):
        super(ZeroConv, self).__init__()    
        self.channels = len(channels_list)
        self.kernel = kernel_size if (isinstance(kernel_size, list) or isinstance(kernel_size, tuple)) else [kernel_size, kernel_size]
        self.padding = padding if (isinstance(padding, list) or isinstance(padding, tuple)) else [padding, padding]
        self.stride = stride if (isinstance(stride, list) or isinstance(stride, tuple)) else [stride, stride]
    
    
    def forward(self, input):
        dvc = input.get_device() if input.is_cuda else 'cpu'
        batch_size = input.shape[0]
        width, height = input.shape[-2:]
        width = self.compute_size(width, self.kernel[0], self.padding[0], self.stride[0])
        height = self.compute_size(height, self.kernel[1], self.padding[1], self.stride[1])
        
        return torch.zeros(torch.Size([batch_size, self.channels, width, height]), device=dvc)
    

    def compute_size(self, dimension, kernel, padding, stride):
        return int( ( (dimension - kernel + 2 * padding) / stride) + 1)


class HintModel(nn.Module):

    def __init__(self, config, teacher, student, device):
        super(HintModel, self).__init__()   
        
        self.hint_layers = nn.ModuleList()
        x = torch.Tensor(1, 3, 416, 416).to(device)
        
        # Get Features sizes
        with torch.no_grad(): 
            _, fts_tch = \
                    teacher(x, config['teacher_indexes']) if type(teacher) is YOLO_Nano \
                    else forward(teacher, x, config['teacher_indexes'])
            _, fts_std = \
                student(x, config['student_indexes']) if type(student) is YOLO_Nano \
                    else forward(student, x, config['student_indexes']) 

        for i, (ft_tch, ft_std) in enumerate(zip(fts_tch, fts_std)):
            _, chnl_tch, w_tch, h_tch = ft_tch.shape
            _, chnl_std, w_std, h_std = ft_std.shape
            if w_tch != w_std or h_tch != h_std: 
                print(f'Skiping {i}-th hint layer because shapes do not match:\n\tTeacher -> {[w_tch, h_tch]}\tStudent -> {[w_std, h_std]}')
            else: 
                hint_layer = nn.Sequential(
                    nn.Conv2d(
                        in_channels=chnl_std, out_channels=chnl_tch,
                        kernel_size=(1, 1), stride=(1, 1),
                        padding=(1, 1) # Missing padding trick from https://arxiv.org/pdf/1507.00448.pdf to match the resolution
                    ),
                    nn.LeakyReLU(0.1, inplace=True)
                )
                self.hint_layers.add_module(f'hint_{i}', hint_layer)
    

    def forward(self, fts):
        y = []
        for i, x in enumerate(fts):
            y.append(self.hint_layers[i](x))
        return y


def forward(model, x, fts_indexes=[], verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out, features = [], [], []
        verbose = False
        if verbose:
            str = ''
            print('0', x.shape)

        for i, (mdef, module) in enumerate(zip(model.module_defs, model.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'softconv', 'multibias', 'multiconv_multibias', 'halfconv', 'inception', 'upsample', 'maxpool']:
                if mtype == 'softconv': 
                    x1 = module[0](x, model.temp, model.ticket)
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
            
            # Features to output
            if i in fts_indexes: features.append(x)
            
            out.append(x if i in model.routs else [])
            if verbose:
                print('%g/%g %s -' % (i, len(model.module_list), mtype), list(x.shape), str)
                str = ''

        if model.training: # train
            return yolo_out, features if len(fts_indexes) else yolo_out
        elif ONNX_EXPORT: # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
                                                                                # scores, boxes: 3780x80, 3780x4
            return x[0], torch.cat(x[1:3], 1), features if len(fts_indexes) else x[0], torch.cat(x[1:3], 1)
        else: # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p, features if len(fts_indexes) else torch.cat(io, 1), p