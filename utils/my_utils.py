def create_train_argparser():
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
    parser.add_argument('--cache_labels', action='store_true', help='cache labels for faster training')
    parser.add_argument('--weights', type=str, help='initial weights')
    parser.add_argument('--arc', type=str, help='yolo architecture')  # default, uCE, uBCE
    parser.add_argument('--darknet', type=str, help='Darknet type (default, multibias)')
    parser.add_argument('--name', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single_cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--var', type=float, help='debug variable')
    # hyp parameters
    parser.add_argument('--lr0', type=float, help='initial learning rate')
    parser.add_argument('--lrf', type=float, help='final learning rate')
    parser.add_argument('--momentum', type=float, help='momentum to Stochastic Gradient Descendent')
    parser.add_argument('--weight_decay', type=float, help='weight decay for pg1 parameters')
    # My additioned parameters
    parser.add_argument('--scheduler', type=str, help='kind of learning rate scheduler')
    parser.add_argument('--decay_steps', type=str)
    parser.add_argument('--exponential_ramp', action='store_true', help="changes inverse exponential learning rate decay to be exponential")
    parser.add_argument('--cosine_ramp', action='store_true', help="changes inverse exponential learning rate decay to be cosine ramp")
    parser.add_argument('--xavier_uniform', action='store_true', help='initialize model with xavier uniform function')
    parser.add_argument('--xavier_norm', action='store_true', help='initialize model with xavier normal function')
    parser.add_argument('--gamma', type=float, help='gamma used in learning rate decay')
    parser.add_argument('--params', type=str, default='params/default.yaml', help='json config to load the hyperparameters')
    parser.add_argument('--seed', type=int, default=0, help='seed to function init_seeds')
    args = vars(parser.parse_args())

    return args


def create_test_argparser():
    import argparse

    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='path to weights file')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save_json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single_cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--mask', action='store_true', help='wheter has a mask inside the checkpoint')
    parser.add_argument('--mask_weight', type=str, default=None, help='wheter mask is another checkpoint')
    parser.add_argument('--architecture', type=str, default='default', help='mask to apply in the model')
    args = vars(parser.parse_args())

    pieces = args['weights'].split('/')
    working_dir = ''
    if len(pieces) > 0:
        for i in range(len(pieces) - 1): # eliminate the last part (*.pt) to take the folder
            working_dir += pieces[i] + '/'
    args['working_dir'] = working_dir

    return args


def create_prune_argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)  # 500200 batches at bs 16, 117263 COCO images = 300 epochs
    parser.add_argument('--batch_size', type=int)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, help='*.cfg path')
    parser.add_argument('--data', type=str, help='*.data path')
    parser.add_argument('--multi_scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img_size', nargs='+', type=int, help='train and test image-sizes')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, help='gsutil bucket')
    parser.add_argument('--cache_images', action='store_true', help='cache images for faster training')
    parser.add_argument('--cache_labels', action='store_true', help='cache labels for faster training')
    parser.add_argument('--weights', type=str, help='initial weights')
    parser.add_argument('--arc', type=str, help='yolo architecture')  # default, uCE, uBCE
    parser.add_argument('--darknet', type=str, help='Darknet type (default, multibias)')
    parser.add_argument('--name', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    # hyp parameters
    parser.add_argument('--lr0', type=float, help='initial learning rate')
    parser.add_argument('--lrf', type=float, help='final learning rate')
    parser.add_argument('--momentum', type=float, help='momentum to Stochastic Gradient Descendent')
    parser.add_argument('--weight_decay', type=float, help='weight decay for pg1 parameters')
    # My additioned parameters
    parser.add_argument('--scheduler', type=str, help='kind of learning rate scheduler')
    parser.add_argument('--decay_steps', type=str)
    parser.add_argument('--exponential_ramp', action='store_true', help="changes inverse exponential learning rate decay to be exponential")
    parser.add_argument('--cosine_ramp', action='store_true', help="changes inverse exponential learning rate decay to be cosine ramp")
    parser.add_argument('--xavier_uniform', action='store_true', help='initialize model with xavier uniform function')
    parser.add_argument('--xavier_norm', action='store_true', help='initialize model with xavier normal function')
    parser.add_argument('--gamma', type=float, help='gamma used in learning rate decay')
    parser.add_argument('--seed', type=int, default=0, help='seed to function init_seeds')
    # Pruning parameters
    parser.add_argument('--iterations', type=int, help='One iteration have X epochs. Prune and reseting at the final of each iteration, except the last')
    parser.add_argument('--reseting', type=int, help='Save backup in each iteration on epoch X for reseting')
    parser.add_argument('--pruning_time', type=int, help='Counter for the number of prunes')
    parser.add_argument('--pruning_rate', type=float, help='Percent of connections to remove')
    parser.add_argument('--prune_kind', type=str, help='Way to perform the prune')
    # Specific Continuous Sparsification parameters
    parser.add_argument('--mask_initial_value', type=float, help='initialization for pseudo-mask s')
    parser.add_argument('--mask_lr', type=float, help='learing rate for pseudo-mask s')
    parser.add_argument('--mask_momentum', type=float, help='momentum for pseudo-mask s')
    parser.add_argument('--final_temperature', type=float, help='final beta to binarize sigmoid function')
    parser.add_argument('--lambda', type=float, help='lambda for L1 mask regularization')

    parser.add_argument('--params', type=str, required=True, help='json config to load the hyperparameters')
    args = vars(parser.parse_args())

    return args


def create_kd_argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)  # 500200 batches at bs 16, 117263 COCO images = 300 epochs
    parser.add_argument('--batch_size', type=int)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, help='batches to accumulate before optimizing')
    parser.add_argument('--teacher_cfg', type=str, help='*.cfg teacher path')
    parser.add_argument('--student_cfg', type=str, help='*.cfg student path')
    parser.add_argument('--data', type=str, help='*.data path')
    parser.add_argument('--multi_scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img_size', nargs='+', type=int, help='train and test image-sizes')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, help='gsutil bucket')
    parser.add_argument('--cache_images', action='store_true', help='cache images for faster training')
    parser.add_argument('--cache_labels', action='store_true', help='cache labels for faster training')
    parser.add_argument('--teacher_weights', type=str, help='initial teacher weights')
    parser.add_argument('--student_weights', type=str, help='initial student weights')
    parser.add_argument('--teacher_arc', type=str, help='yolo architecture')  # default, uCE, uBCE
    parser.add_argument('--student_arc', type=str, help='yolo architecture')  # default, uCE, uBCE
    parser.add_argument('--teacher_darknet', type=str, help='Teacher Darknet type (default, multibias)')
    parser.add_argument('--student_darknet', type=str, help='Student Darknet type (default, multibias)')
    parser.add_argument('--name', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    # hyp parameters
    parser.add_argument('--lr0', type=float, help='initial learning rate')
    parser.add_argument('--lrf', type=float, help='final learning rate')
    parser.add_argument('--momentum', type=float, help='momentum to Stochastic Gradient Descendent')
    parser.add_argument('--weight_decay', type=float, help='weight decay for pg1 parameters')
    # My additioned parameters
    parser.add_argument('--scheduler', type=str, help='kind of learning rate scheduler')
    parser.add_argument('--decay_steps', type=str)
    parser.add_argument('--exponential_ramp', action='store_true', help="changes inverse exponential learning rate decay to be exponential")
    parser.add_argument('--cosine_ramp', action='store_true', help="changes inverse exponential learning rate decay to be cosine ramp")
    parser.add_argument('--xavier_uniform', action='store_true', help='initialize model with xavier uniform function')
    parser.add_argument('--xavier_norm', action='store_true', help='initialize model with xavier normal function')
    parser.add_argument('--gamma', type=float, help='gamma used in learning rate decay')
    parser.add_argument('--seed', type=int, default=0, help='seed to function init_seeds')
    # Teacher parameters
    parser.add_argument('--mask', action='store_true', help='There is a mask to load inside teacher checkpoint')
    parser.add_argument('--mask_path', type=str, help='There is a mask to load on another path')
    # KD parameters
    parser.add_argument('--params', type=str, default='params/KD_Guobin.yaml', help='json config to load the hyperparameters')
    parser.add_argument('--mu', type=float, help='Weight the hard and soft classification loss')
    parser.add_argument('--ni', type=float, help='Weight the teacher bounded regression loss. Default value specified by authors')
    parser.add_argument('--margin', type=float, help='Student need to have a bbox loss < bbox_t + margin')
    parser.add_argument('--cls_function', type=str, help='Fuction to apply in predictions before Soft Classification Loss')
    args = vars(parser.parse_args())

    return args


#############
# From ONet #
#############
def load_config(path, default_path=None):
    import yaml
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
#############
# From ONet #
#############


def create_config(opt):
    import time
    import os 

    config = load_config(opt['params'])

    # Update config with opt (terminal arguments)
    for key, value in opt.items():
        if value is not None:
            if key in config['hyp']: config['hyp'][key] = value
            else: config[key] = value

    # Create sub_working_dir
    if opt['resume']:
        if 'student_weights' in config:
            if config['student_weights']: folders = config['student_weights'].split('/')
        else: 
            if config['weights']: folders = config['weights'].split('/')
            else:
                print("No weights right to resume")
                exit()
        config['sub_working_dir'] = ''
        for i in range(len(folders) - 1):
            config['sub_working_dir'] += folders[i] + '/'
    else:
        sub_working_dir = '{}/{}/size-{}/{}'.format(
            config['working_dir'],
            config['cfg'].split('/')[1].split('.')[0],
            config['img_size'][0] if opt['multi_scale'] is False and opt['img_size'] is None 
                else 'multi_scale' if opt['multi_scale'] is True else opt['img_size'][0],

            '{}_{}_{}/{}_{}_{}/'.format(
                time.strftime("%Y", time.localtime()),
                time.strftime("%m", time.localtime()),
                time.strftime("%d", time.localtime()),
                time.strftime("%H", time.localtime()),
                time.strftime("%M", time.localtime()),
                time.strftime("%S", time.localtime())
            )
        )
        if not os.path.exists(sub_working_dir):
            os.makedirs(sub_working_dir)
        config["sub_working_dir"] = sub_working_dir

    return config


def create_scheduler(opt, optimizer, start_epoch):
    import math
    import torch.optim.lr_scheduler as lr_scheduler

    if opt['scheduler'] == 'multi-step':
        values = opt['decay_steps'].split(' ')
        try:
            milestones = [int(x) for x in values]
        except:
            milestones = [round(opt['epochs'] * float(x)) for x in values]

        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= milestones, gamma= opt['gamma'])
    elif opt['scheduler'] == 'lambda':
        if opt['exponential_ramp']:
            lf = lambda x: 10 ** (opt['hyp']['lrf'] * x / opt['epochs'])  # exp ramp
        elif opt['cosine_ramp']:
            lf = lambda x: (1 + math.cos(x * math.pi / opt['epochs'])) / 2 * 0.99 + 0.01  # cosine https://arxiv.org/pdf/1812.01187.pdf
        else:
            lf = lambda x: 1 - 10 ** (opt['hyp']['lrf'] * (1 - x / opt['epochs']))  # inverse exp ramp
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch

    return scheduler


def create_optimizer(model, opt):
    import torch.optim as optim

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'mask' not in k:
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


def initialize_model(model, function):
    
    for name, param in model.named_parameters():
        if 'BatchNorm2d' not in name and 'bias' not in name:
            function(param)


def create_dataloaders(config):
    import os
    from torch.utils.data import DataLoader
    from utils.parse_config import parse_data_cfg
    from utils.datasets import LoadImagesAndLabels

    data = config['data']
    img_size, img_size_test = config['img_size'] if len(config['img_size']) == 2 else config['img_size'] * 2  # train, test sizes
    batch_size = config['batch_size']

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    valid_path = data_dict['valid']

    # Dataset
    dataset = LoadImagesAndLabels(
        train_path, img_size, batch_size,
        augment=True, hyp=config['hyp'],  cache_labels=config['cache_labels'],# augmentation hyperparameters
        cache_images=config['cache_images'],
    )

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    trainloader = DataLoader(
        dataset, batch_size = batch_size, num_workers = nw,
        pin_memory = True, collate_fn = dataset.collate_fn
    )

    # Testloader
    validloader = DataLoader(
        LoadImagesAndLabels(
            valid_path, img_size_test, batch_size * 2,
            hyp = config['hyp'], rect = True, cache_labels = config['cache_labels'],
            cache_images = False
        ),
        batch_size = batch_size * 2, num_workers = nw, pin_memory = True, collate_fn = dataset.collate_fn
    )

    return trainloader, validloader


def load_checkpoints(config, model, optimizer, device, try_download_function, darknet_load_function):
    import torch
    
    start_epoch = 0
    best_fitness = 0.0
    try_download_function(config['weights'])
    if config['weights'].endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(config['weights'], map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (config['weights'], config['cfg'], config['weights'])
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(config['results_file'], 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt
        torch.cuda.empty_cache()

    elif len(config['weights']) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        darknet_load_function(model, config['weights'])

    return start_epoch, best_fitness, model, optimizer


def load_checkpoints_mask(config, model, mask, optimizer, device, try_download_function, darknet_load_function):
    import torch
    
    start_epoch = 0
    start_iteration = 0
    best_fitness = 0.0
    try_download_function(config['weights'])
    if config['weights'].endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(config['weights'], map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (config['weights'], config['cfg'], config['weights'])
            raise KeyError(s) from e

        try:
            if chkpt['mask'] is not None:
                mask.load_state_dict(chkpt['mask'])
        except:
            print('Mask not found')

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(config['results_file'], 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        try:
            start_iteration = chkpt['iteration']
        except:
            print('Iteration not found. initializin on 0')
            start_iteration = 0
        start_epoch = chkpt['epoch'] + 1
        del chkpt
        torch.cuda.empty_cache()

    elif len(config['weights']) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        darknet_load_function(model, config['weights'])

    return start_iteration, start_epoch, best_fitness, model, mask, optimizer


def load_kd_checkpoints(config, teacher, student, mask, hint_model, optimizer, device):
    import torch
    
    start_epoch = 0
    best_fitness = 0.0
    
    # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
    chkpt = torch.load(config['teacher_weights'], map_location=device)

    # load teacher
    try:
        chkpt['model'] = {k: v for k, v in chkpt['model'].items() if teacher.state_dict()[k].numel() == v.numel()}
        teacher.load_state_dict(chkpt['model'], strict=False)
    except KeyError as e:
        s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
            "See https://github.com/ultralytics/yolov3/issues/657" % (config['weights'], config['cfg'], config['weights'])
        raise KeyError(s) from e
    # load mask
    try:
        if 'mask' is chkpt:
            mask.load_state_dict(chkpt['mask'])
        elif config['mask_path'] is not None:
            msk = torch.load(config['mask_path'], map_location=device)
            if 'mask' in msk: mask.load_state_dict(msk['mask'])
            else: mask.load_state_dict(msk)
            del msk
    except:
        print('Mask not found')
    
    # Reseting checkpoint
    del chkpt
    torch.cuda.empty_cache()
    if config['student_weights'].endswith('.pt'):
        chkpt = torch.load(config['student_weights'], map_location=device)
        
        # load student
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if student.state_dict()[k].numel() == v.numel()}
            student.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (config['weights'], config['cfg'], config['weights'])
            raise KeyError(s) from e
        
        # load hint models
        try:
            chkpt['hint'] = {k: v for k, v in chkpt['hint'].items() if hint_model.state_dict()[k].numel() == v.numel()}
            hint_model.load_state_dict(chkpt['hint'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (config['weights'], config['cfg'], config['weights'])
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(config['results_file'], 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt
        torch.cuda.empty_cache()

    return start_epoch, best_fitness, teacher, student, mask, hint_model, optimizer