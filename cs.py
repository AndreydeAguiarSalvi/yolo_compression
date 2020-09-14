# Look in this paper to see the SPP: Spatial Pyramidal Pooling
# https://arxiv.org/pdf/1903.08589.pdf

import torch.distributed as dist

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.my_utils import create_prune_argparser, create_config, create_scheduler, create_optimizer, initialize_model, create_dataloaders, load_checkpoints_mask
from utils.pruning import sum_of_the_weights

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed


def compute_remaining_weights(masks):
    return 1 - sum(float((m == 0).sum()) for m in masks) / sum(m.numel() for m in masks)

def adjust_learning_rate(optimizer, value):
    for param_group in optimizer.param_groups: param_group['lr'] = value

counter = 0

def train(iteration, best_fitness, prebias, trainloader, validloader, config, scheduler, mask_scheduler, optimizer, mask_optim, tb_writer):
    config['last'] = config['sub_working_dir'] + 'last_it_{}.pt'.format(iteration)
    config['best'] = config['sub_working_dir'] + 'best_it_{}.pt'.format(iteration)
    max_wo_best = 0
    ###############
    # Start epoch #
    ###############
    for epoch in range(start_epoch, config['epochs']):  
        model.train()
        model.gr = 1 - (1 + math.cos(min(epoch * 2, config['epochs']) * math.pi / config['epochs'])) / 2  # GIoU <-> 1.0 loss ratio
        
        # if mask_scheduler is not None: mask_scheduler.step()
        # if mask_optim is not None: 
        #     if epoch == 0:
        #         adjust_learning_rate(mask_optim, config['mask_lr'])
        #     elif epoch == int(.65 * config['epochs']):
        #         adjust_learning_rate(mask_optim, config['mask_lr'] * .1) # 65% and 84% of 150, respectivelly, as in the paper (56/85 and 71/85)
        #     elif epoch == int(.84 * config['epochs']): 
        #         adjust_learning_rate(mask_optim, config['mask_lr'] * .01) # 65% and 84% of 150, respectivelly, as in the paper (56/85 and 71/85)

        # Prebias
        if prebias:
            ne = max(round(30 / nb), 3)  # number of prebias epochs
            ps = np.interp(epoch, [0, ne], [0.1, config['hyp']['lr0'] * 2]), \
                np.interp(epoch, [0, ne], [0.9, config['hyp']['momentum']])  # prebias settings (lr=0.1, momentum=0.9)
            if epoch == ne:
                print_model_biases(model)
                prebias = False

            # Bias optimizer settings
            optimizer.param_groups[2]['lr'] = ps[0]
            if optimizer.param_groups[2].get('momentum') is not None:  # for SGD but not Adam
                optimizer.param_groups[2]['momentum'] = ps[1]

        # Update image weights (optional)
        if trainloader.dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(trainloader.dataset.labels, nc=nc, class_weights=w)
            trainloader.dataset.indices = random.choices(range(trainloader.dataset.n), weights=image_weights, k=trainloader.dataset.n)  # rand weighted idx

        ###########
        # From CS #
        ###########
        if epoch > 0: model.temp *= temp_increase
        if iteration == 0 and epoch == config['reseting']: model.checkpoint()

        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 9) % ('Iter', 'Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(trainloader), total=nb)  # progress bar
        ####################
        # Start mini-batch #
        ####################
        for i, (imgs, targets, paths, _) in pbar: 
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Plot images with bounding boxes
            if ni < 1:
                f = config['sub_working_dir'] + 'train_batch%g.png' % i  # filename
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer:
                    tb_writer.add_image(f, cv2.imread(f)[:, :, ::-1], dataformats='HWC')

            # Multi-Scale training
            if config['multi_scale']:
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            masks = [m.mask for m in model.mask_modules]
            entries_sum = sum(m.sum() for m in masks)
            loss += config['lambda'] * entries_sum

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if mask_optim is not None: 
                    mask_optim.step()
                    mask_optim.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 3 + '%10.3g' * 6) % ('%g/%g' % (iteration, config['iterations']-1), '%g/%g' % (epoch, config['epochs'] - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)
        ##################
        # End mini-batch #
        ##################

        # Update scheduler
        scheduler.step()
        if mask_scheduler is not None: mask_scheduler.step()

        final_epoch = epoch + 1 == config['epochs']
        if not config['notest'] or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test.test(
                cfg = cfg, data = data, batch_size=batch_size,
                img_size=img_size_test, model=model, 
                conf_thres=0.001,  # 0.001 if opt.evolve or (final_epoch and is_coco) else 0.01,
                iou_thres=0.6, save_json=final_epoch and is_coco,
                dataloader=validloader, folder = config['sub_working_dir']
            )    

        # Write epoch results
        with open(config['results_file'], 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(config['name']) and config['bucket']:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (config['bucket'], config['name']))

        # Write Tensorboard results
        if tb_writer:
            global counter
            x = list(mloss) + list(results)
            titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, counter)
            counter += 1

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi
            max_wo_best = 0
        else:
            max_wo_best += 1
            if config['earlie_stop'] and max_wo_best == config['earlie_stop']: print('Ending training due to early stop')

        # Save training results
        save = (not config['nosave']) or (final_epoch and not config['evolve'])
        if save:
            with open(config['results_file'], 'r') as f:
                # Create checkpoint
                chkpt = {'iteration': iteration,
                        'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, config['last'])

            # Save best checkpoint
            if best_fitness == fi:
                torch.save(chkpt, config['best'])

            # Delete checkpoint
            del chkpt
            torch.cuda.empty_cache()
        
        if config['earlie_stop'] and max_wo_best == config['earlie_stop']: break
    #############
    # End epoch #
    #############
    try: 
        print(f'Iteration {iteration} finished with {compute_remaining_weights(masks)} remaining weights.')
    except:
        x = torch.Tensor(1, 3, 416, 416).to(device)
        y = model(x)
        masks = [m.mask for m in model.mask_modules]
        print(f'Iteration {iteration} finished with {compute_remaining_weights(masks)} remaining weights.')
        del x 
        del y



if __name__ == '__main__':
    args = create_prune_argparser()
    config = create_config(args)
    print("sub working dir: %s" % config['sub_working_dir'])

    # Saving configurations
    import json
    with open(config['sub_working_dir'] + 'config.json', 'w') as f:
        json.dump(config, f)
    f.close()

    config['last'] = config['weights'] if 'last' in config['weights'] else config['sub_working_dir'] + 'last.pt'
    config['best'] = config['weights'].replace('last', 'best') if 'last' in config['weights'] else config['sub_working_dir'] + 'best.pt'
    config['results_file'] = config['sub_working_dir'] + 'results.txt'
    config['weights'] = config['last'] if config['resume'] else config['weights']

    print(config)
    
    device = torch_utils.select_device(config['device'], apex=mixed_precision, batch_size=config['batch_size'])
    if device.type == 'cpu':
        mixed_precision = False

    tb_writer = None
    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir= config['sub_working_dir'] + 'runs/')
    except:
        pass

    ####################
    # Start Old Train 1#
    ####################
    cfg = config['cfg']
    data = config['data']
    img_size, img_size_test = config['img_size'] if len(config['img_size']) == 2 else config['img_size'] * 2  # train, test sizes
    batch_size = config['batch_size']
    accumulate = config['accumulate']  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = config['weights']  # initial training weights

    # Initialize
    init_seeds(config['seed'])
    if config['multi_scale']:
        img_sz_min = round(img_size / 32 / 1.5)
        img_sz_max = round(img_size / 32 * 1.5)
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = SoftDarknet(cfg, arc=config['arc']).to(device)
    optimizer = create_optimizer(model, config)

    start_epoch = 0
    best_fitness = 0.0
    
    start_iteration, start_epoch, best_fitness, model, _, optimizer = load_checkpoints_mask(
        config, model, None,  
        optimizer, device, 
        attempt_download, load_darknet_weights
    )

    if config['xavier_norm']:
        initialize_model(model, torch.nn.init.xavier_normal_)
    elif config['xavier_uniform']:
        initialize_model(model, torch.nn.init.xavier_uniform_)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    trainloader, validloader = create_dataloaders(config)

    # Start training
    nb = len(trainloader)
    prebias = start_epoch == 0
    model.nc = nc  # attach number of classes to model
    model.arc = config['arc']  # attach yolo architecture
    model.hyp = config['hyp']  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(trainloader.dataset.labels, nc).to(device)  # attach class weights
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    print('Starting training for %g epochs...' % config['epochs'])
    ###################
    # End Old Train 1 #
    ###################

    mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask' in p[0], model.named_parameters()))
    mask_optim = torch.optim.SGD(mask_params, lr=config['mask_lr'], momentum=config['mask_momentum'], nesterov=True)
    # mask_scheduler = create_scheduler(config, mask_optim, start_epoch)
    
    model.ticket = False
    config['epochs'] = int(config['epochs'] / config['iterations'])
    iters_per_reset = config['epochs']-1
    temp_increase = config['final_temperature']**(1./iters_per_reset)
    for it in range(start_iteration, config['iterations']):
        scheduler = create_scheduler(config, optimizer, start_epoch)
        mask_scheduler = create_scheduler(config, mask_optim, start_epoch)
        train(it, best_fitness, prebias, trainloader, validloader, config, scheduler, mask_scheduler, optimizer, mask_optim, tb_writer) 
        start_epoch = 0
        best_fitness = .0
        model.temp = 1
        if it != config['iterations']-1: model.prune()
    
    mask_optim = None
    model.ticket = True
    model.rewind_weights()
    optimizer = create_optimizer(model, config)
    config['epochs'] = int(config['epochs'] * config['iterations'])
    scheduler = create_scheduler(config, optimizer, start_epoch)
    best_fitness = .0
    train(it+1, best_fitness, prebias, trainloader, validloader, config, scheduler, None, optimizer, mask_optim, tb_writer)

    #####################
    # Start Old Train 2 #
    #####################
    # Without the os.rename on the last results
    if not config['evolve']:
        plot_results(folder= config['sub_working_dir'])

    print('%g epochs completed in %.3f hours.\n' % (config['epochs'] - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    ###################
    # End Old Train 2 #
    ###################