import os


def train_argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch_size', type=int)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, help='*.cfg path')
    parser.add_argument('--data', type=str, help='*.data path')
    parser.add_argument('--multi_scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img_size', nargs='+', type=int, help='train and test image-sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, help='gsutil bucket')
    parser.add_argument('--cache_images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, help='initial weights')
    parser.add_argument('--arc', type=str, help='yolo architecture')  # default, uCE, uBCE
    parser.add_argument('--name', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single_cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--var', type=float, help='debug variable')
    parser.add_argument('--scheduler', type=str, help='kind of learning rate scheduler')
    parser.add_argument('--decay_steps', type=str)
    parser.add_argument('--gamma', type=float, help='gamma used in learning rate decay')
    parser.add_argument('--params', type=str, default='params/default.json', help='json config to load the hyperparameters')
    args = vars(parser.parse_args())

    return args


def test_argparser():
    import argparse

    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save_json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single_cls', action='store_true', help='train as single-class dataset')
    args = vars(parser.parse_args())

    pieces = args['weights'].split('/')
    working_dir = ''
    if len(pieces) > 0:
        for i in range(len(pieces) - 1): # eliminate the last part (*.pt) to take the folder
            working_dir += pieces[i] + '/'
    args['working_dir'] = working_dir

    return args


def create_config(opt):
    import json
    import time 

    json_file = open(opt['params'])
    json_str = json_file.read()
    config = json.loads(json_str)

    # Create sub_working_dir
    sub_working_dir = '{}/{}/size-{}/{}'.format(
        config['working_dir'],
        opt['cfg'].split('/')[1].split('.')[0], # Get the architecture name
        config['img_size'][0] if opt['multi_scale'] is False and opt['img_size'] is None 
            else 'multi_scale' if opt['multi_scale'] is True else opt['img_size'][0],

        '{}_{}_{}/{}_{}/'.format(
            time.strftime("%Y", time.localtime()),
            time.strftime("%m", time.localtime()),
            time.strftime("%d", time.localtime()),
            time.strftime("%H", time.localtime()),
            time.strftime("%M", time.localtime())
        )
    )
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir

    for key, value in opt.items():
        if value is not None:
            config[key] = value

    return config


def create_scheduler(opt, optimizer, start_epoch):
    import torch.optim.lr_scheduler as lr_scheduler

    if opt['scheduler'] == 'multi-step':
        milestones = [round(opt['epochs'] * float(x)) for x in opt['decay_steps'].split(' ')]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= milestones, gamma= opt['gamma'])
    elif opt['scheduler'] == 'lambda':
        # lf = lambda x: 10 ** (opt['hyp']['lrf'] * x / epochs)  # exp ramp
        lf = lambda x: 1 - 10 ** (opt['hyp']['lrf'] * (1 - x / opt['epochs']))  # inverse exp ramp
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1

    return scheduler


def create_optimizer(model, opt):
    import torch.optim as optim

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt['adam']:
        # opt['hyp']['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=opt['hyp']['lr0'])
        # optimizer = AdaBound(pg0, lr=opt['hyp']['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=opt['hyp']['lr0'], momentum=opt['hyp']['momentum'], nesterov=True)
    
    optimizer.add_param_group({'params': pg1, 'weight_decay': opt['hyp']['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    # https://github.com/alphadl/lookahead.pytorch
    # optimizer = torch_utils.Lookahead(optimizer, k=5, alpha=0.5)

    return optimizer