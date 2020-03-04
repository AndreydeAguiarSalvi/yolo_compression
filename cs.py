# Look in this paper to see the SPP: Spatial Pyramidal Pooling
# https://arxiv.org/pdf/1903.08589.pdf

import torch.distributed as dist

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.my_utils import create_prune_argparser, create_config, create_scheduler, create_optimizer, initialize_model, create_dataloaders, load_checkpoints
from utils.pruning import sum_of_the_weights, create_backup, rewind_weights, apply_mask_CS, create_mask_CS, CS, compute_mask


def train():
    cfg = config['cfg']
    img_size, img_size_test = config['img_size'] if len(config['img_size']) == 2 else config['img_size'] * 2  # train, test sizes
    epochs = config['epochs']  # 500200 batches at bs 64, 117263 images = 273 epochs
    accumulate = config['accumulate']  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = config['weights']  # initial training weights

    # Initialize
    init_seeds()
    if config['multi_scale']:
        img_sz_min = round(img_size / 32 / 1.5)
        img_sz_max = round(img_size / 32 * 1.5)
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(config['data'])
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg, arc=config['arc']).to(device)
    mask = create_mask_CS(model, init_value = config['mask_initial_value']).to(device)

    optimizer = create_optimizer(model, config)
    start_epoch, best_fitness, model, weights, optimizer = load_checkpoints(
        config, model, weights, 
        optimizer, device, 
        attempt_download, load_darknet_weights
    )
    scheduler = create_scheduler(config, optimizer, start_epoch)

    # Kind of initialization
    if config['xavier_norm']:
        initialize_model(model, torch.nn.init.xavier_normal_)
    elif config['xavier_uniform']:
        initialize_model(model, torch.nn.init.xavier_uniform_)


    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
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
    print('Starting training for %g epochs...' % epochs)

    ###########
    # From CS #
    ###########
    iters_per_reset = config['epochs']-1
    temp_increase = config['final_temperature']**(1./iters_per_reset)
    ticket = False
    mask_optim = torch.optim.SGD(mask.parameters(), lr=config['mask_lr'], momentum=config['mask_momentum'], nesterov = True)
    mask_scheduler = create_scheduler(config, mask_optim, start_epoch)
    beta = 1

    counter = 0
    ###################
    # Start Iteration #
    ###################
    for it in range(config['iterations']):
        
        if it == config['iterations'] - 1: # In original code, this step is outter from outer_round loop
            ticket = True
            mask = mask.to('cpu')
            print('Rewind weights.')
            backup = backup.to(device)
            rewind_weights(model, backup)
            backup = backup.to('cpu')
            mask = mask.to(device)

            optimizer = create_optimizer(model, config)
            start_epoch = 0
            scheduler = create_scheduler(config, optimizer, start_epoch)
        ###############
        # Start epoch #
        ###############
        for epoch in range(start_epoch, epochs):  
            model.train()

            # Prebias
            if prebias:
                if epoch < 3:  # prebias
                    ps = np.interp(epoch, [0, 3], [0.1, config['hyp']['lr0']]), 0.0  # prebias settings (lr=0.1, momentum=0.0)
                else:  # normal training
                    ps = config['hyp']['lr0'], config['hyp']['momentum']  # normal training settings
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

            mloss = torch.zeros(4).to(device)  # mean losses
            print(('\n' + '%10s' * 9) % ('Iter', 'Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(enumerate(trainloader), total=nb)  # progress bar

            # CS #
            if epoch > 0:
                beta *= temp_increase
            if epoch == config['reseting'] - 1 and it == 0:
                backup = create_backup(model).to('cpu')

            ####################
            # Start mini-batch #
            ####################
            for i, (imgs, targets, paths, _) in pbar: 
                ni = i + nb * epoch  # number integrated batches (since train start)
                
                ##############
                # Apply mask #
                ##############
                pseudo_mask = compute_mask(mask, config['mask_initial_value'], beta, ticket)
                apply_mask_CS(model, pseudo_mask)

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

                if it == 0 and epoch == 0 and i == 0:
                    tb_writer.add_graph( model, imgs)

                # Compute loss
                loss, loss_items = compute_loss(pred, targets, model, not prebias)
                ######
                # CS #
                ######
                entries_sum = sum_of_the_weights(pseudo_mask)
                if not torch.isfinite(entries_sum):
                    print('WARNING: non-finite entries sum, ending training ')
                    return results
                loss += config['lambda'] * entries_sum

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Scale loss by nominal batch_size of 64
                loss *= config['batch_size'] / 64

                # Compute gradient
                loss.backward()

                if it != config['iterations'] - 1: # The last train is outer from outer_round loop, without mask optimizer and scheduler
                    # Maybe we cannot accumulate loss in mask, 
                    # because the loss mask is computed over
                    # the last pseudo_mask, and probally we do
                    # not have backward matrix in mask, once time
                    # mask is updated just with L1 regularization 
                    mask_optim.step()
                    mask_optim.zero_grad()

                # Optimize accumulated gradient
                if ni % accumulate == 0:
                    for key in mask:
                        tb_writer.add_histogram( mask[key].grad, ni)
                    optimizer.step()
                    optimizer.zero_grad()

                # Print batch results
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 3 + '%10.3g' * 6) % ('%g/%g' % (it, config['iterations']-1), '%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
                pbar.set_description(s)
            ##################
            # End mini-batch #
            ##################

            print(f"last entries_sum in loss: {config['lambda'] * entries_sum}")

            # Update scheduler
            scheduler.step()
            if it != config['iterations'] - 1: # The last train is outer from outer_round loop, without mask optimizer and scheduler
                mask_scheduler.step() 
            
            final_epoch = epoch + 1 == epochs
            if not config['notest'] or final_epoch:  # Calculate mAP
                is_coco = any([x in config['data'] for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
                # Apply mask before test
                compute_mask(mask, config['mask_initial_value'], beta, ticket)
                apply_mask_CS(model, mask)
                results, maps = test.test(
                    cfg = cfg, data = config['data'], batch_size=config['batch_size'] * 2,
                    img_size= img_size_test, model=model, 
                    conf_thres=1E-3 if config['evolve'] or (final_epoch and is_coco) else 0.1,  # 0.1 faster
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

            # Save training results
            save = (not config['nosave']) or (final_epoch and not config['evolve'])
            if save:
                with open(config['results_file'], 'r') as f:
                    # Create checkpoint
                    chkpt = {'epoch': epoch,
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
        #############
        # End epoch #
        #############

        # Saving current mask before prune
        torch.save(mask.state_dict(), config['sub_working_dir'] + 'mask_{}_{}.pt'.format(
                config['pruning_time'], 'prune' if config['pruning_time'] == 1 else 'prunes'
            )
        )
        # Saving current model before prune
        torch.save(model.state_dict(), config['sub_working_dir'] + 'model_it_{}.pt'.format(it+1))

        if it < config['iterations'] - 1:
            beta = 1
        
        if it < config['iterations'] - 2: # In original code, it not prunes in iteration rounds-1, and train more one iteration after the for()
            print('Applying Continuous Sparsification')
            CS(mask, config['mask_initial_value'], beta)
            config['pruning_time'] += 1    

    
    #################
    # End Iteration #
    #################

    n = config['name']
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, 'last%s.pt' % n, 'best%s.pt' % n
        os.rename(config['results_file'], config['sub_working_dir'] + fresults)
        os.rename(config['last'], config['sub_working_dir'] + flast) if os.path.exists(config['last']) else None
        os.rename(config['best'], config['sub_working_dir'] + fbest) if os.path.exists(config['best']) else None
        # Updating results, last and best
        config['results_file'] = config['sub_working_dir'] + fresults
        config['last'] = config['sub_working_dir'] + flast
        config['best'] = config['sub_working_dir'] + fbest

        if config['bucket']:  # save to cloud
            os.system('gsutil cp %s gs://%s/results' % (fresults, config['bucket']))
            os.system('gsutil cp %s gs://%s/weights' % (config['sub_working_dir'] + flast, config['bucket']))
            # os.system('gsutil cp %s gs://%s/weights' % (config['sub_working_dir'] + fbest, config['bucket']))

    if not config['evolve']:
        plot_results(folder= config['sub_working_dir'])

    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    print(f'Final model with {sum_of_the_weights(mask)} weights')
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


if __name__ == '__main__':
    args = create_prune_argparser()
    config = create_config(args)
    print("sub working dir: %s" % config['sub_working_dir'])

    # Saving configurations
    import json
    with open(config['sub_working_dir'] + 'config.json', 'w') as f:
        json.dump(config, f)
    f.close()

    config['last'] = config['sub_working_dir'] + 'last.pt'
    config['best'] = config['sub_working_dir'] + 'best.pt'
    config['results_file'] = config['sub_working_dir'] + 'results.txt'
    config['weights'] = config['last'] if config['resume'] else config['weights']

    print(config)
    
    device = torch_utils.select_device(config['device'], batch_size=config['batch_size'])

    tb_writer = None
    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir= config['sub_working_dir'] + 'runs/')
    except:
        pass

    train()  # train normally
