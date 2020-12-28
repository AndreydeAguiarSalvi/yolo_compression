import cv2
import torch
import argparse
import numpy as np
from models import *
from utils.utils import compute_loss
from torch.utils.data import DataLoader
import torchvision.transforms as Transforms
from utils.torch_utils import select_device
from utils.gradcam import show_cam_on_image
from utils.datasets import LoadImagesAndLabels

hyp = {
    'giou': 3.54,
    'cls': 37.4,
    'cls_pw': 1.0,
    'obj': 64.3,
    'obj_pw': 1.0,
    'iou_t': 0.2
}

gradients = []

def save_(grad):
    global gradients
    gradients.append(grad)


def YOLO_Gradcam(model, dataloader, device, args):
    l = len(dataloader)
    j = 0
    model.eval()
    for i, (imgs, labels, paths, _) in enumerate(dataloader):
        imgs = imgs.to(device).float() / 255.0 
        labels = labels.to(device)

        for img, path in tqdm(zip(imgs, paths)):
            # One (image, bboxes) per time #
            img = torch.stack([img])
            id = labels[:, 0] == j
            _, y_hat, fts = model(imgs, [model.yolo_layers[args['head']]-1])
            j += 1

            # Saving features
            fts[0].register_hook(save_)

            # Computing loss and backward
            loss, _ = compute_loss(y_hat, labels[id], model)
            model.zero_grad()
            loss.backward(retain_graph=True)

            ###########
            # Gradcam #
            ###########
            # getting grandients and features
            grads_val = gradients[0]
            target = fts[0]
            target = target[0, :]
            # weighting gradients in cam
            weights = torch.mean(grads_val, axis=(2, 3))[0, :]
            cam = torch.zeros(target.shape[1:], device=device, dtype=torch.float32)
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]
            # creating the mask
            cam = cam = torch.where(cam > 0, cam, torch.tensor(0., device=device))
            resize = Transforms.Compose([ Transforms.ToPILImage(), Transforms.Resize(img.shape[2:]), Transforms.ToTensor() ])
            cam = resize(torch.stack([cam.cpu()]))[0] # torch resizes only 3D or moreD tensors, not 2D
            cam = cam - torch.min(cam)
            mask = cam / torch.max(cam)
            
            # creating a name to grad image
            ext = path.split('.')[-1]
            name = path.split(os.sep)[-1].split('.')[0]
            grad_name = f"{args['output']}{os.sep}{name}_{args['head']}_{'all'}.{ext}"            
            orig_name = f"{args['output']}{os.sep}{name}.{ext}"
            # Saving results
            img = cv2.cvtColor(img[0].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            show_cam_on_image(img, mask, grad_name)
            cv2.imwrite(orig_name, np.uint8(255 * img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/pascal/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='weights/voc_yolov3/size-multi_scale/2020_03_16/13_19_34/model_it_1.pt', help='path to weights file')
    parser.add_argument('--data', type=str, default='data/voc2012.data', help='.data pointing images to load')
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--batch_size', type=int, default=8, help='number of images per mini-batch')
    parser.add_argument('--rect', action='store_true', help='rectangular inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--head', type=int, default=0, help='YOLO head to evaluate')
    args = vars(parser.parse_args())
    print(args)

    device = select_device(args['device'], apex=False, batch_size=args['batch_size'])

    #########
    # Model #
    #########
    if 'nano' in args['cfg']: model = YOLO_Nano(args['cfg']).to(device)
    else: model = Darknet(args['cfg']).to(device)
    model.hyp = hyp
    model.nc = 20 if 'voc' in args['data'] else 12
    model.arc = 'default'
    model.gr = 1 - (1 + math.cos(math.pi)) / 2  # GIoU <-> 1.0 loss ratio
    # Load args['weights']
    attempt_download(args['weights'])
    if args['weights'].endswith('.pt'):  # pytorch format
        try:
            model.load_state_dict(torch.load(args['weights'], map_location=device)['model'])
        except:
            model.load_state_dict(torch.load(args['weights'], map_location=device))
    else:  # darknet format
        load_darknet_weights(model, args['weights'])

    ########
    # Data #
    ########
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

    if not os.path.exists(args['output']):
        os.makedirs(args['output'])

    YOLO_Gradcam(model, dataloader, device, args)