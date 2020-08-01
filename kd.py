# Based on 
# https://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf

import torch.distributed as dist

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.my_utils import create_kd_argparser, create_config, create_scheduler, create_optimizer, initialize_model, create_dataloaders, load_kd_checkpoints
from utils.pruning import create_mask_LTH, apply_mask_LTH

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed


def train():
    data = config['data']
    img_size, img_size_test = config['img_size'] if len(config['img_size']) == 2 else config['img_size'] * 2  # train, test sizes
    epochs = config['epochs']  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = config['batch_size']
    accumulate = config['accumulate']  # effective bs = batch_size * accumulate = 16 * 4 = 64
    
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

    # Initialize Teacher
    if config['teacher_darknet'] == 'default':
        teacher = Darknet(cfg=config['teacher_cfg'], arc=config['teacher_arc']).to(device)
    elif config['teacher_darknet'] == 'soft':
        teacher = SoftDarknet(cfg=config['teacher_cfg'], arc=config['teacher_arc']).to(device)
    # Initialize Student
    if config['student_darknet'] == 'default':
        student = Darknet(cfg=config['student_cfg'], arc=config['student_arc']).to(device)
    elif config['student_darknet'] == 'soft':
        student = SoftDarknet(cfg=config['student_cfg'], arc=config['student_arc']).to(device)
    # Create Hint Layers
    hint_models = None
    if len(config['teacher_indexes']):
        hint_models = HintModel(config, teacher, student).to(device)
    
    optimizer = create_optimizer(student, config)
    if len(config['teacher_indexes']):
        optimizer.add_param_group({"params": hint_models.parameters()})

    mask = None
    if config['mask'] or config['mask_path'] is not None:
        mask = create_mask_LTH(teacher).to(device)

    start_epoch, best_fitness, teacher, student, mask, hint_models, optimizer = load_kd_checkpoints(
        config, 
        teacher, student, 
        mask, hint_models,
        optimizer, device
    )

    if mask is not None:
        apply_mask_LTH(teacher, mask)
        del mask
        torch.cuda.empty_cache()

    if config['xavier_norm']:
        initialize_model(student, torch.nn.init.xavier_normal_)
    elif config['xavier_uniform']:
        initialize_model(student, torch.nn.init.xavier_uniform_)

    scheduler = create_scheduler(config, optimizer, start_epoch)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        student, optimizer = amp.initialize(student, optimizer, opt_level='O1', verbosity=0)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        teacher = torch.nn.parallel.DistributedDataParallel(teacher, find_unused_parameters=True)
        teacher.yolo_layers = teacher.module.yolo_layers  # move yolo layer indices to top level
        student = torch.nn.parallel.DistributedDataParallel(student, find_unused_parameters=True)
        student.yolo_layers = student.module.yolo_layers  # move yolo layer indices to top level

    trainloader, validloader = create_dataloaders(config)

    # Start training
    nb = len(trainloader)
    prebias = start_epoch == 0
    student.nc = nc  # attach number of classes to student
    teacher.nc = nc
    
    student.arc = config['student_arc']  # attach yolo architecture
    teacher.arc = config['teacher_arc']

    student.hyp = config['hyp']  # attach hyperparameters to student
    teacher.hyp = config['hyp']
    
    student.class_weights = labels_to_class_weights(trainloader.dataset.labels, nc).to(device)  # attach class weights
    teacher.class_weights = student.class_weights

    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    torch_utils.model_info(student, report='summary')  # 'full' or 'summary'
    print('Starting training for %g epochs...' % epochs)

    teacher.train()
    ###############
    # Start epoch #
    ###############
    for epoch in range(start_epoch, epochs):  
        student.train()
        student.gr = 1 - (1 + math.cos(min(epoch * 2, epochs) * math.pi / epochs)) / 2  # GIoU <-> 1.0 loss ratio

        # Prebias
        if prebias:
            ne = max(round(30 / nb), 3)  # number of prebias epochs
            ps = np.interp(epoch, [0, ne], [0.1, config['hyp']['lr0'] * 2]), \
                np.interp(epoch, [0, ne], [0.9, config['hyp']['momentum']])  # prebias settings (lr=0.1, momentum=0.9)
            if epoch == ne:
                print_model_biases(student)
                prebias = False

            # Bias optimizer settings
            optimizer.param_groups[2]['lr'] = ps[0]
            if optimizer.param_groups[2].get('momentum') is not None:  # for SGD but not Adam
                optimizer.param_groups[2]['momentum'] = ps[1]

        # Update image weights (optional)
        if trainloader.dataset.image_weights:
            w = student.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(trainloader.dataset.labels, nc=nc, class_weights=w)
            trainloader.dataset.indices = random.choices(range(trainloader.dataset.n), weights=image_weights, k=trainloader.dataset.n)  # rand weighted idx

        mloss = torch.zeros(5).to(device)  # mean losses
        print(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'hint', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(trainloader), total=nb)  # progress bar
        ####################
        # Start mini-batch #
        ####################
        for i, (imgs, targets, paths, _) in pbar: 
        # for i, (imgs, targets, paths, _) in enumerate(trainloader): 
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

            # Run teacher
            with torch.no_grad():
                if len(config['teacher_indexes']):
                    pred_tch, fts_tch = teacher(imgs, config['teacher_indexes'])
                else: pred_tch = teacher(imgs)
            # Run student
            if len(config['student_indexes']):
                pred_std, fts_std = student(imgs, config['student_indexes'])
            else: pred_std = student(imgs)

            if len(config['teacher_indexes']):
                fts_guided = hint_models(fts_std)

            # Compute loss
            if len(config['teacher_indexes']):
                loss, loss_items = compute_kd_loss(pred_tch, pred_std, targets, fts_tch, fts_guided, teacher, student)
            else:
                loss, loss_items = compute_kd_loss(pred_tch, pred_std, targets, [], [], teacher, student)
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

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 7) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)
        ##################
        # End mini-batch #
        ##################

        # Update scheduler
        scheduler.step()
        
        final_epoch = epoch + 1 == epochs
        if not config['notest'] or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and student.nc == 80
            results, maps = test.test(
                cfg = config['cfg'], data = data, batch_size=batch_size,
                img_size=img_size_test, model=student, 
                conf_thres=0.001,  # 0.001 if opt.evolve or (final_epoch and is_coco) else 0.01,
                iou_thres=0.6, save_json=final_epoch and is_coco, single_cls=config['single_cls'],
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
            titles = ['GIoU', 'Objectness', 'Classification', 'Hint', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save training results
        save = (not config['nosave']) or (final_epoch and not config['evolve'])
        if save:
            with open(config['results_file'], 'r') as f:
                # Create checkpoint
                chkpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': f.read(),
                    'model': student.module.state_dict() if type(student) is nn.parallel.DistributedDataParallel 
                        else student.state_dict(),
                    'hint': None if hint_models is None
                        else hint_models.module.state_dict() if type(hint_models) is nn.parallel.DistributedDataParallel 
                        else hint_models.state_dict(),
                    'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, config['last'])

            # Save best checkpoint
            if best_fitness == fi:
                torch.save(chkpt, config['best'])

            # Delete checkpoint
            del chkpt
            torch.cuda.empty_cache()
    #############
    # End epoch #
    #############

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
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


if __name__ == '__main__':
    args = create_kd_argparser()
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

    print(config)
    
    device = torch_utils.select_device(config['device'], apex=mixed_precision, batch_size=config['batch_size'])
    if device.type == 'cpu':
        mixed_precision = False

    tb_writer = None
    if not config['evolve']:  # Train normally
        try:
            # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir= config['sub_working_dir'] + 'runs/')
        except:
            pass

        train()  # train normally

    else:  # Evolve hyperparameters (optional)
        config['notest'], config['nosave'] = True, True  # only test/save final epoch
        if config['bucket']:
            os.system('gsutil cp gs://%s/evolve.txt .' % config['bucket'])  # download evolve.txt if exists

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
                for i, k in enumerate(config['hyp'].keys()):  # plt.hist(v.ravel(), 300)
                    config['hyp'][k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                config['hyp'][k] = np.clip(config['hyp'][k], v[0], v[1])

            # Train mutation
            results = train()

            # Write mutation results
            print_mutation(config['hyp'], results, config['bucket'])

            # Plot results
            # plot_evolution_results(config['hyp'])
