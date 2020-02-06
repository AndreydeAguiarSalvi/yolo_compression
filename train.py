# Look in this paper to see the SPP: Spatial Pyramidal Pooling
# https://arxiv.org/pdf/1903.08589.pdf

import torch.distributed as dist

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.my_utils import create_argparser, create_config, create_scheduler, create_optimizer

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed


def train():
    cfg = opt['cfg']
    data = opt['data']
    img_size, img_size_test = opt['img_size'] if len(opt['img_size']) == 2 else opt['img_size'] * 2  # train, test sizes
    epochs = opt['epochs']  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt['batch_size']
    accumulate = opt['accumulate']  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt['weights']  # initial training weights

    # Initialize
    init_seeds()
    if opt['multi_scale']:
        img_sz_min = round(img_size / 32 / 1.5)
        img_sz_max = round(img_size / 32 * 1.5)
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt['single_cls'] else int(data_dict['classes'])  # number of classes

    # Remove previous results
    ##############################
    # Maybe remove in the future #
    ##############################
    for f in glob.glob('*_batch*.png') + glob.glob(opt['results_file']):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg, arc=opt['arc']).to(device)

    optimizer = create_optimizer(model, opt)

    start_epoch = 0
    best_fitness = 0.0
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt['weights'], opt['cfg'], opt['weights'])
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(opt['results_file'], 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)

    scheduler = create_scheduler(opt, optimizer, start_epoch)

    # # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=opt['hyp'],  # augmentation hyperparameters
                                  rect=opt['rect'],  # rectangular training
                                  cache_labels=True,
                                  cache_images=opt['cache_images'],
                                  single_cls=opt['single_cls'])

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt['rect'],  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, img_size_test, batch_size * 2,
                                                                 hyp=opt['hyp'],
                                                                 rect=True,
                                                                 cache_labels=True,
                                                                 cache_images=opt['cache_images'],
                                                                 single_cls=opt['single_cls']),
                                             batch_size=batch_size * 2,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Start training
    nb = len(dataloader)
    prebias = start_epoch == 0
    model.nc = nc  # attach number of classes to model
    model.arc = opt['arc']  # attach yolo architecture
    model.hyp = opt['hyp']  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    ###############
    # Start epoch #
    ###############
    for epoch in range(start_epoch, epochs):  
        model.train()

        # Prebias
        if prebias:
            if epoch < 3:  # prebias
                ps = 0.1, 0.9  # prebias settings (lr=0.1, momentum=0.9)
            else:  # normal training
                ps = opt['hyp']['lr0'], opt['hyp']['momentum']  # normal training settings
                print_model_biases(model)
                prebias = False

            # Bias optimizer settings
            optimizer.param_groups[2]['lr'] = ps[0]
            if optimizer.param_groups[2].get('momentum') is not None:  # for SGD but not Adam
                optimizer.param_groups[2]['momentum'] = ps[1]

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        ####################
        # Start mini-batch #
        ####################
        for i, (imgs, targets, paths, _) in pbar: 
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Plot images with bounding boxes
            if ni == 0:
                fname = 'train_batch%g.png' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

            # Multi-Scale training
            if opt.multi_scale:
                if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model, not prebias)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)
        ##################
        # End mini-batch #
        ##################

        final_epoch = epoch + 1 == epochs
        if not opt['notest'] or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size * 2,
                                      img_size=img_size_test,
                                      model=model,
                                      conf_thres=1E-3 if opt['evolve'] or (final_epoch and is_coco) else 0.1,  # 0.1 faster
                                      iou_thres=0.6,
                                      save_json=final_epoch and is_coco,
                                      single_cls=opt['single_cls'],
                                      dataloader=testloader)

        # Update scheduler
        scheduler.step()

        # Write epoch results
        with open(opt['results_file'], 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt['name']) and opt['bucket']:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt['bucket'], opt['name']))

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results)
            titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save training results
        save = (not opt['nosave']) or (final_epoch and not opt['evolve'])
        if save:
            with open(opt['results_file'], 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, opt['last'])

            # Save best checkpoint
            if best_fitness == fi:
                torch.save(chkpt, opt['best'])

            # Save backup every 10 epochs (optional)
            # if epoch > 0 and epoch % 10 == 0:
            #     torch.save(chkpt, opt['sub_working_dir'] + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt
    #############
    # End epoch #
    #############

    n = opt['name']
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, 'last%s.pt' % n, 'best%s.pt' % n
        os.rename(opt['results_file'], opt['sub_working_dir'] + fresults)
        os.rename(opt['last'], opt['sub_working_dir'] + flast) if os.path.exists(opt['last']) else None
        os.rename(opt['best'], opt['sub_working_dir'] + fbest) if os.path.exists(opt['best']) else None
        # Updating results, last and best
        opt['results_file'] = opt['sub_working_dir'] + fresults
        opt['last'] = opt['sub_working_dir'] + flast
        opt['best'] = opt['sub_working_dir'] + fbest

        if opt['bucket']:  # save to cloud
            os.system('gsutil cp %s gs://%s/results' % (fresults, opt['bucket']))
            os.system('gsutil cp %s gs://%s/weights' % (opt['sub_working_dir'] + flast, opt['bucket']))
            # os.system('gsutil cp %s gs://%s/weights' % (opt['sub_working_dir'] + fbest, opt['bucket']))

    if not opt['evolve']:
        plot_results(name= opt['sub_working_dir'] + 'results.png')

    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


if __name__ == '__main__':
    opt = create_argparser()
    opt = create_config(opt)
    print("sub working dir: %s" % opt['sub_working_dir'])

    opt['last'] = opt['sub_working_dir'] + 'last.pt'
    opt['best'] = opt['sub_working_dir'] + 'best.pt'
    opt['results_file'] = opt['sub_working_dir'] + 'results.txt'
    opt['weights'] = opt['last'] if opt['resume'] else opt['weights']

    print(opt)
    
    device = torch_utils.select_device(opt['device'], apex=mixed_precision, batch_size=opt['batch_size'])
    if device.type == 'cpu':
        mixed_precision = False

    tb_writer = None
    if not opt['evolve']:  # Train normally
        try:
            # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
            from torch.utils.tensorboard import SummaryWriter
            ######################
            # Save in the future #
            ######################
            tb_writer = SummaryWriter(logdir= opt['sub_working_dir'] + 'runs/')
        except:
            pass

        train()  # train normally

    else:  # Evolve hyperparameters (optional)
        opt['notest'], opt['nosave'] = True, True  # only test/save final epoch
        if opt['bucket']:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt['bucket'])  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                if method == 1:
                    v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(opt['hyp'].keys()):  # plt.hist(v.ravel(), 300)
                    opt['hyp'][k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                opt['hyp'][k] = np.clip(opt['hyp'][k], v[0], v[1])

            # Train mutation
            results = train()

            # Write mutation results
            print_mutation(opt['hyp'], results, opt['bucket'])

            # Plot results
            # plot_evolution_results(opt['hyp'])
