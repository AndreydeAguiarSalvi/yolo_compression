import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *
from utils.my_utils import create_test_argparser


def compute_removed_weights(masks):
    return sum(float((m == 0).sum()) for m in masks)


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         model=None,
         dataloader=None,
         folder='',
         mask=None,
         mask_weight=None,
         architecture='default'):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(args['device'], batch_size=batch_size)
        verbose = args['task'] == 'test'

        # Remove previous
        for f in glob.glob(folder + 'test_batch*.png'):
            os.remove(f)

        # Initialize model
        if 'soft' in cfg:
            model = SoftDarknet(cfg=cfg).to(device)
            model.ticket = True

            x = torch.Tensor(1, 3, 416, 416).to(device)
            y = model(x)
            masks = [m.mask for m in model.mask_modules]
            print(f"Evaluating model with {compute_removed_weights(masks)} parameters removed.")
        else:
            model = Darknet(cfg=cfg).to(device)

        if mask or mask_weight:
            from utils.pruning import sum_of_the_weights, apply_mask_LTH, create_mask_LTH
            msk = create_mask_LTH(model)
            initial_weights = sum_of_the_weights(msk)
            if mask: msk.load_state_dict(torch.load(weights, map_location=device)['mask'])
            else: msk.load_state_dict(torch.load(mask_weight, map_location=device))
            final_weights = sum_of_the_weights(msk)
            apply_mask_LTH(model, msk)
            print(f'Evaluating model with initial weights number of {initial_weights} and final of {final_weights}. \nReduction of {final_weights * 100. / initial_weights}%.')
            del msk

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            try:
                try:
                    model.load_state_dict(torch.load(weights, map_location=device)['model'])
                except:
                    model.load_state_dict(torch.load(weights, map_location=device))
            except:
                load_from_old_version( model, torch.load(weights, map_location=device) )
        else:  # darknet format
            load_darknet_weights(model, weights)

        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['test'] if 'test' in data else data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, img_size, batch_size, rect=True, single_cls=single_cls, cache_labels=True)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Plot images with bounding boxes
        f = folder + 'test_batch%g.png' % batch_i  # filename
        if batch_i < 1 and not os.path.exists(f):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)


        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(imgs)  # inference and training outputs
            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(d[5])],
                                  'bbox': [floatn(x, 3) for x in box[di]],
                                  'score': floatn(d[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))
    
    # Saving the average evaluations
    class_results = open(folder + 'per_class_evaluations.txt', 'w')
    print(s)
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1), file=class_results)
    for i, c in enumerate(ap_class):
        # Saving the evaluations per class
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]), file=class_results)
    # Closing the evaluations .txt
    class_results.close()

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open(folder + 'results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except:
            print('WARNING: missing pycocotools package, can not compute official COCO mAP. See requirements.txt.')

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO(glob.glob('../COCO2014/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes(folder + 'results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)

    # Print speeds
    if verbose:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (img_size, img_size, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    args = create_test_argparser()
    args['save_json'] = args['save_json'] or any([x in args['data'] for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    print(args)

    if args['task'] == 'test':  # test normally
        test(
                cfg = args['cfg'], data = args['data'], weights = args['weights'],
                batch_size = args['batch_size'], img_size = args['img_size'], conf_thres = args['conf_thres'],
                iou_thres = args['iou_thres'], save_json = args['save_json'], folder = args['working_dir'],
                mask = args['mask'], mask_weight = args['mask_weight'], architecture = args['architecture']
            )

    elif args['task'] == 'benchmark': # mAPs at 320-608 at conf 0.5 and 0.7
        y = []
        for i in [320, 416, 512, 608]: # img-size
            for j in [0.5, 0.7]: # iou-thres
                t = time.time()
                r = test(
                        cfg = args['cfg'], data = args['data'], weights = args['weights'], 
                        batch_size = args['batch_size'], img_size = i, conf_thres = args['conf_thres'], 
                        iou_thres = j, save_json = args['save_json'], folder = args['working_dir'],
                        mask = args['mask'], mask_weight = args['mask_weight'], architecture = args['architecture']
                    )[0]
                y.append(r + (time.time() - t,))
        np.savetxt(args['working_dir'] + 'benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

    elif args['task'] == 'study': # Parameter study
        y = []
        x = np.arange(0.4, 0.9, 0.05) # iou-thres
        for i in x:
            t = time.time()
            r = test(
                cfg = args['cfg'], data = args['data'], weights = args['weights'], 
                batch_size = args['batch_size'], img_size = args['img_size'], conf_thres = args['conf_thres'], 
                iou_thres = i, save_json = args['save_json'], folder = args['working_dir'],
                mask = args['mask'], mask_weight = args['mask_weight'], architecture = args['architecture']
            )[0]
            y.append(r + (time.time() - t,))
        np.savetxt(args['working_dir'] + 'study.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

        # Plot
        fig, ax = plt.subplots(3, 1, figsize=(6, 6))
        y = np.stack(y, 0)
        ax[0].plot(x, y[:, 2], marker='.', label='mAP@0.5')
        ax[0].set_ylabel('mAP')
        ax[1].plot(x, y[:, 3], marker='.', label='mAP@0.5:0.95')
        ax[1].set_ylabel('mAP')
        ax[2].plot(x, y[:, -1], marker='.', label='time')
        ax[2].set_ylabel('time (s)')
        for i in range(3):
            ax[i].legend()
            ax[i].set_xlabel('iou_thr')
        fig.tight_layout()
        plt.savefig(args['working_dir'] + 'study.jpg', dpi=200)
