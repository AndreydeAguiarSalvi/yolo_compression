# Based on 
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9046859
# Adapted with
# https://github.com/soumith/ganhacks

import torch.distributed as dist
import random
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

ft = torch.cuda.FloatTensor

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
    # Create Discriminators
    D_models = None
    if len(config['teacher_indexes']):
        D_models = Discriminator(config['teacher_indexes'], teacher, config['D_kernel_size'], False).to(device)
    
    G_optim = create_optimizer(student, config)
    D_optim = create_optimizer(D_models, config, is_D=True)
    GAN_criterion = torch.nn.BCEWithLogitsLoss()

    mask = None
    if ('mask' in config and config['mask']) or ('mask_path' in config and config['mask_path']):
        print('Creating mask')
        mask = create_mask_LTH(teacher).to(device)

    start_epoch, best_fitness, teacher, student, mask, D_models, G_optim, D_optim = load_kd_checkpoints(
        config, 
        teacher, student, 
        mask, D_models,
        G_optim, D_optim, device
    )

    if mask is not None:
        print('Applying mask in teacher')
        apply_mask_LTH(teacher, mask)
        del mask
        torch.cuda.empty_cache()

    if config['xavier_norm']:
        initialize_model(student, torch.nn.init.xavier_normal_)
    elif config['xavier_uniform']:
        initialize_model(student, torch.nn.init.xavier_uniform_)

    G_scheduler = create_scheduler(config, G_optim, start_epoch)
    D_scheduler = create_scheduler(config, D_optim, start_epoch)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        student, G_optim = amp.initialize(student, G_scheduler, opt_level='O1', verbosity=0)

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
    max_wo_best = 0
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
            G_optim.param_groups[2]['lr'] = ps[0]
            if G_optim.param_groups[2].get('momentum') is not None:  # for SGD but not Adam
                G_optim.param_groups[2]['momentum'] = ps[1]

        # Update image weights (optional)
        if trainloader.dataset.image_weights:
            w = student.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(trainloader.dataset.labels, nc=nc, class_weights=w)
            trainloader.dataset.indices = random.choices(range(trainloader.dataset.n), weights=image_weights, k=trainloader.dataset.n)  # rand weighted idx

        mloss = torch.zeros(9).to(device)  # mean losses
        print(('\n' + '%10s' * 13) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'G_loss', 'D_loss', 'D_x', 'D_g_z1', 'D_g_z2', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(trainloader), total=nb)  # progress bar
        ####################
        # Start mini-batch #
        ####################
        for i, (imgs, targets, paths, _) in pbar: 
            real_data_label = ft(imgs.shape[0], device=device).uniform_(.7, 1.0)
            fake_data_label = ft(imgs.shape[0], device=device).uniform_(.0, .3)

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

            # Run student
            if len(config['student_indexes']) and epoch < config['second_stage']:
                pred_std, fts_std = student(imgs, config['student_indexes'])
            else: pred_std = student(imgs)

            
            ###################################################
            # Update D: maximize log(D(x)) + log(1 - D(G(z))) #
            ###################################################
            D_loss_real, D_loss_fake, D_x, D_g_z1 = ft([.0]), ft([.0]), ft([.0]), ft([.0])
            if epoch < config['second_stage']:
                # Run teacher
                with torch.no_grad():
                    _, fts_tch = teacher(imgs, config['teacher_indexes'])
                
                # Adding noise to Discriminator: flipping labels
                if random.random() < .05:
                    aux = real_data_label
                    real_data_label = fake_data_label
                    fake_data_label = aux

                # Discriminate the real data
                real_data_discrimination = D_models(fts_tch)
                for output in real_data_discrimination: D_x += output.mean().item() / 3.
                # Discriminate the fake data
                fake_data_discrimination = D_models([x.detach() for x in fts_std])
                for output in fake_data_discrimination: D_g_z1 += output.mean().item() / 3.
                
                # Compute loss
                for x in real_data_discrimination:
                    D_loss_real += GAN_criterion(x.view(-1), real_data_label)
                for x in fake_data_discrimination:
                    D_loss_fake += GAN_criterion(x.view(-1), fake_data_label)

                # Scale loss by nominal batch_size of 64
                D_loss_real *= batch_size / 64
                D_loss_fake *= batch_size / 64

                # Compute gradient
                D_loss_real.backward()
                D_loss_fake.backward()

                # Optimize accumulated gradient
                if ni % accumulate == 0:
                    D_optim.step()
                    D_optim.zero_grad()

            ###################################
            # Update G: maximize log(D(G(z))) #
            ###################################
            G_loss, D_g_z2 = ft([.0]), ft([.0])
            if epoch < config['second_stage']:
                # Since we already update D, perform another forward with fake batch through D
                fake_data_discrimination = D_models(fts_std)
                for output in fake_data_discrimination: D_g_z2 += output.mean().item() / 3.
                
                # Compute loss
                real_data_label = torch.ones(imgs.shape[0], device=device)
                for x in fake_data_discrimination:
                    G_loss += GAN_criterion(x.view(-1), real_data_label) # fake labels are real for generator cost
                obj_detec_loss, loss_items = ft([.0]), ft([.0, .0, .0, .0])
                
                # Scale loss by nominal batch_size of 64
                G_loss *= batch_size / 64
                
                # Compute gradient
                G_loss.backward()

            else:
                # Compute loss
                obj_detec_loss, loss_items = compute_loss(pred_std, targets, student)
        
                # Scale loss by nominal batch_size of 64
                obj_detec_loss *= batch_size / 64

                # Compute gradient
                obj_detec_loss.backward()

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                G_optim.step()
                G_optim.zero_grad()

            D_loss = D_loss_real + D_loss_fake
            total_loss = obj_detec_loss + D_loss + G_loss + obj_detec_loss
            all_losses = torch.cat( [loss_items[:3], G_loss, D_loss, D_x, D_g_z1, D_g_z2, total_loss] ).detach() 
            if not torch.isfinite(total_loss):
                print('WARNING: non-finite loss, ending training ', all_losses)
                return results

            # Print batch results
            mloss = (mloss * i + all_losses) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 11) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)
        ##################
        # End mini-batch #
        ##################

        # Update scheduler
        G_scheduler.step()
        D_scheduler.step()
        
        final_epoch = epoch + 1 == epochs
        if not config['notest'] or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and student.nc == 80
            thres = .1
            while True:
                try:
                    results, maps = test.test(
                        cfg = config['cfg'], data = data, batch_size=1,
                        img_size=img_size_test, model=student, 
                        conf_thres=thres if epoch < config['second_stage'] else 0.001,
                        iou_thres=0.6, save_json=final_epoch and is_coco, single_cls=config['single_cls'],
                        dataloader=None, folder = config['sub_working_dir']
                    )
                    break
                except:
                    thres += .1    

        # Write epoch results
        with open(config['results_file'], 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(config['name']) and config['bucket']:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (config['bucket'], config['name']))

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results)
            titles = [
                'GIoU', 'Objectness', 'Classification', 'Generator Loss', 'Discriminator Loss', 
                'D_x', 'D_g_z1', 'D_g_z2' 'Train Loss',
                'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
            ]
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi
            max_wo_best = 0
        else:
            max_wo_best += 1
            if config['early_stop'] and max_wo_best == config['early_stop']: print('Ending training due to early stop')

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
                    'D': D_models.state_dict(),
                    'G_optim': None if final_epoch else G_optim.state_dict(),
                    'D_optim': None if final_epoch else D_optim.state_dict()     
                }

            # Save last checkpoint
            torch.save(chkpt, config['last'])

            # Save best checkpoint
            if best_fitness == fi:
                torch.save(chkpt, config['best_gan'] if epoch < config['second_stage'] else config['best'])

            # Delete checkpoint
            del chkpt
            torch.cuda.empty_cache()
        
        if config['early_stop'] and max_wo_best == config['early_stop']: break
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
    config['best_gan'] = config['sub_working_dir'] + 'best_gan.pt'
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