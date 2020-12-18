import os
import math
import copy
import argparse
from models import *
from utils.utils import *
from utils.datasets import *
from torch.utils.data import DataLoader
from utils.torch_utils import select_device
from utils.gradcam import GradCam, GuidedBackpropReLUModel, show_cam_on_image, deprocess_image


def compute_grad(model, dataloader, args):
    
    if args['layer']: layer = [args['layer']]
    else: layer = [model.yolo_layers[args['head']]-1]
    grad_cam = GradCam(model, layer)
    gb_model = GuidedBackpropReLUModel(copy.deepcopy(model))

    for imgs, labels, paths, _ in tqdm(dataloader):
        imgs = imgs.to(device).float() / 255.0 
        labels = labels.to(device)

        for i, (img, path) in enumerate(zip(imgs, paths)):
            x = torch.stack([img])
            ###########
            # Gradcam #
            ###########
            target = None
            if args['class']: target = args['class']
            elif args['h_obj']:
                l = labels[labels[:, 0] == i, :]
                area = l[:, 4] * l[:, 5]
                target = l[torch.argmax(area), 1]

            mask = grad_cam(x, args['head'], args['anchor'], target)
            # names
            ext = path.split('.')[-1]
            name = path.split(os.sep)[-1].split('.')[0]
            grad_name = f"{args['output']}{os.sep}_grad_{name}_{args['head']}_{args['anchor']}.{ext}"
            orig_name = f"{args['output']}{os.sep}{name}.{ext}"
            # Saving results
            x_ = cv2.cvtColor(x[0].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            show_cam_on_image(x_, mask, grad_name)
            cv2.imwrite(orig_name, np.uint8(255 * x_))

            ##################
            # GuidedBackprop #
            ##################
            x.requires_grad = True
            gb = gb_model(x, args['head'], args['anchor'], target)
            gb = gb.transpose((1, 2, 0))
            cam_mask = cv2.merge([mask, mask, mask])
            cam_gb = deprocess_image(cam_mask*gb)
            gb = deprocess_image(gb)
            # Saving results
            gb_name = f"{args['output']}{os.sep}_gb_{name}_{args['head']}_{args['anchor']}.{ext}"
            cam_name = f"{args['output']}{os.sep}_cam-gb_{name}_{args['head']}_{args['anchor']}.{ext}"
            cv2.imwrite(gb_name, gb)
            cv2.imwrite(cam_name, cam_gb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/pascal/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='weights/voc_yolov3/size-multi_scale/2020_03_16/13_19_34/model_it_1.pt', help='path to weights file')
    parser.add_argument('--data', type=str, default='data/small_voc.data', help='.data pointing images to load')
    parser.add_argument('--source', type=str, default='', help='source')  # input file/folder
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--batch_size', type=int, default=16, help='number of images per mini-batch')
    parser.add_argument('--rect', action='store_true', help='rectangular inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--head', type=int, default=0, help='YOLO head to evaluate')
    parser.add_argument('--anchor', type=int, default=0, help='YOLO anchor to evaluate')
    parser.add_argument('--layer', type=int, default=None, help='layer to get the features and create the gradcam. If None, the gradcam is created over the self pre-YOLO Head layer')
    parser.add_argument('--class', type=int, default=None, help='Class to evaluate the features. If None, the hightest prediction will be used')
    parser.add_argument('--h_obj', action='store_true', help='if True, it uses the class from the bigger object in the scene.')
    args = vars(parser.parse_args())
    print(args)

    device = select_device(args['device'], apex=False, batch_size=args['batch_size'])

    #########
    # Model #
    #########
    model = Darknet(args['cfg']).to(device)
    # Load args['weights']
    attempt_download(args['weights'])
    if args['weights'].endswith('.pt'):  # pytorch format
        try:
            try:
                model.load_state_dict(torch.load(args['weights'], map_location=device)['model'])
            except:
                model.load_state_dict(torch.load(args['weights'], map_location=device))
        except:
            load_from_old_version( model, torch.load(args['weights'], map_location=device) )
    else:  # darknet format
        load_darknet_weights(model, args['weights'])

    ########
    # Data #
    ########
    if args['source']:
        dataloader = LoadImages(args['source'], img_size=args['img_size'])
    elif args['data']:
        data = parse_data_cfg(args['data'])
        path = data['test'] if 'test' in data else data['valid']  # path to test images
        dataset = LoadImagesAndLabels(
            path, args['img_size'], args['batch_size'], rect=args['rect'], 
            single_cls= int(data['classes']) == True, cache_labels=True
        )

        dataloader = DataLoader(
            dataset, batch_size=args['batch_size'], shuffle=False,
            num_workers=min([os.cpu_count(), args['batch_size'], 8]),
            pin_memory=True, collate_fn=dataset.collate_fn
        )

    subf = str(args['layer']) if args['layer'] else 'bigger' if args['h_obj'] else 'self'
    if not os.path.exists(args['output'] + os.sep + subf):
        os.makedirs(args['output'] + os.sep + subf)
    args['output'] += os.sep + subf
    
    compute_grad(model, dataloader, args)
