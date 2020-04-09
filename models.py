# TODO
# concluir função de Loss para KD
# refatorar cs.py
# fazer kd.py
# 

# Maybe help to understand this fucking code:
# https://www.cyberailab.com/home/a-closer-look-at-yolov3
# https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
import torch.nn.functional as F

from utils.google_utils import *
from utils.parse_config import *
from utils.utils import *

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
                ))
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


class Inception(nn.Module):
    # from https://github.com/sanghoon/pytorch_imagenet/blob/master/models/pvanet.py
    def __init__(self, n_in, n_out, in_stride=1, preAct=False, lastAct=True, proj=False):
        super(Inception, self).__init__()

        # Config
        self._preAct = preAct
        self._lastAct = lastAct
        self.n_in = n_in
        self.n_out = n_out
        self.act_func = nn.ReLU
        self.act = F.relu
        self.in_stride = in_stride

        self.n_branches = 0
        self.n_outs = []        # number of output feature for each branch

        self.proj = nn.Conv2d(n_in, n_out, 1, stride=in_stride) if proj else None

    def add_branch(self, module, n_out):
        # Create branch
        br_name = 'branch_{}'.format(self.n_branches)
        setattr(self, br_name, module)

        # Last output chns.
        self.n_outs.append(n_out)

        self.n_branches += 1

    def branch(self, idx):
        br_name = 'branch_{}'.format(idx)
        return getattr(self, br_name, None)

    def add_convs(self, n_kernels, n_chns):
        assert(len(n_kernels) == len(n_chns))

        n_last = self.n_in
        layers = []

        stride = -1
        for k, n_out in zip(n_kernels, n_chns):
            if stride == -1:
                stride = self.in_stride
            else:
                stride = 1

            # Initialize params
            conv = nn.Conv2d(n_last, n_out, kernel_size=k, bias=False, padding=int(k / 2), stride=stride)
            bn = nn.BatchNorm2d(n_out)

            # Instantiate network
            layers.append(conv)
            layers.append(bn)
            layers.append(self.act_func())

            n_last = n_out

        self.add_branch(nn.Sequential(*layers), n_last)

        return self

    def add_poolconv(self, kernel, n_out, type='MAX'):

        assert(type in ['AVE', 'MAX'])

        n_last = self.n_in
        layers = []

        # Pooling
        if type == 'MAX':
            layers.append(nn.MaxPool2d(kernel, padding=int(kernel/2), stride=self.in_stride))
        elif type == 'AVE':
            layers.append(nn.AvgPool2d(kernel, padding=int(kernel/2), stride=self.in_stride))

        # Conv - BN - Act
        layers.append(nn.Conv2d(n_last, n_out, kernel_size=1))
        layers.append(nn.BatchNorm2d(n_out))
        layers.append(self.act_func())

        self.add_branch(nn.Sequential(*layers), n_out)

        return self


    def finalize(self):
        # Add 1x1 convolution
        total_outs = sum(self.n_outs)

        self.last_conv = nn.Conv2d(total_outs, self.n_out, kernel_size=1)
        self.last_bn = nn.BatchNorm2d(self.n_out)

        return self

    def forward(self, x):
        x_sc = x

        if (self._preAct):
            x = self.act(x)

        # Compute branches
        h = []
        for i in range(self.n_branches):
            module = self.branch(i)
            assert(module != None)

            h.append(module(x))

        x = torch.cat(h, dim=1)

        x = self.last_conv(x)
        x = self.last_bn(x)

        if (self._lastAct):
            x = self.act(x)

        if (x_sc.get_device() != x.get_device()):
            print("Something's wrong")

        # Projection
        if self.proj:
            x_sc = self.proj(x_sc)

        x = x + x_sc

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
        scaling = 1. / F.sigmoid(self.mask_initial_value)
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
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules: m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.load_state_dict(m.checkpoint)
                
    def prune(self):
        for m in self.mask_modules: m.prune(self.temp)


class SoftDarknet(MaskedNet):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        super(MaskedNet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, arc)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

        self.temp = 1
        self.mask_modules = [m for m in self.modules() if type(m) == SoftMaskedConv2d]

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
                if mtype == 'softconv': x = module(x, self.temp, self.ticket)
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


class YOLO_Teacher_Student(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        super(YOLO_Teacher_Student, fts_index, self).__init__()

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
            
            if i == fts_index: yolo_out.append(x)

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

    def forward(self, x):
        loss = 0
        yolo_outputs = []
        image_size = x.size(2)
        current_size = x.shape[-2:]

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pep1(out)
        out = self.ep1(out)
        out = self.pep2(out)
        out = self.pep3(out)
        out = self.ep2(out)
        out = self.pep4(out)
        out = self.conv3(out)
        out = self.fca1(out)
        out = self.pep5(out)
        out = self.pep6(out)
        
        out_pep7 = self.pep7(out)
        out = self.ep3(out_pep7)
        out = self.pep8(out)
        out = self.pep9(out)
        out = self.pep10(out)
        out = self.pep11(out)
        out = self.pep12(out)
        out = self.pep13(out)
        out = self.pep14(out)

        out_pep15 = self.pep15(out)
        out = self.ep4(out_pep15)
        out = self.pep16(out)
        out = self.conv4(out)
        out = self.ep5(out)
        out = self.pep17(out)

        out_conv5 = self.conv5(out)
        out = F.interpolate(self.conv6(out_conv5), scale_factor=2)
        out = torch.cat([out, out_pep15], dim=1)
        out = self.pep18(out)
        out = self.pep19(out)
        
        out_conv7 = self.conv7(out)
        out = F.interpolate(self.conv8(out_conv7), scale_factor=2)
        out = torch.cat([out, out_pep7], dim=1)
        out = self.pep20(out)
        out = self.pep21(out)
        out = self.pep22(out)
        out_conv9 = self.conv9(out)
        yolo_outputs.append(self.yolo_layer52(out_conv9, current_size))

        out = self.ep6(out_conv7)
        out_conv10 = self.conv10(out)
        yolo_outputs.append(self.yolo_layer26(out_conv10, current_size))

        out = self.ep7(out_conv5)
        out_conv11 = self.conv11(out)
        yolo_outputs.append(self.yolo_layer13(out_conv11, current_size))

        if self.training: # train
            return yolo_outputs
        elif ONNX_EXPORT: # export
            x = [torch.cat(x, 0) for x in zip(*yolo_outputs)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else: # test
            io, p = zip(*yolo_outputs)  # inference output, training output
            return torch.cat(io, 1), p
    

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