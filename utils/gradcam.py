# Based on https://github.com/jacobgil/pytorch-grad-cam
# The original code does't work with YOLO
# This is an adaptation from https://github.com/jacobgil/pytorch-grad-cam to work with YOLO
# The original code assumes that the models have two functions: features and classifier

import cv2
import torch
import numpy as np
from torch.autograd import Function
import torchvision.transforms as Transforms


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def get_gradients(self):
        return self.gradients

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        _, train_out, fts = self.model(x, self.target_layers)
        
        for item in fts:
            item.register_hook(self.save_gradient)

        return fts, train_out

def show_cam_on_image(img, mask, path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(path, np.uint8(255 * cam))

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.device = next(iter(model.parameters())).device
        self.extractor = ModelOutputs(self.model, target_layer)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, head=0, anchor=0, index=None):
        features, output = self.extractor(input)

        if index == None:           # [3times [bs, anchors, grid, grid, xywh + classes] ]
            indexes = torch.nonzero( output[head] == torch.max(output[head][0, anchor, ..., 5:]) )
            id_a, id_b, id_c, id_d, id_e = indexes[0]
        else:
            indexes = torch.nonzero( output[head] == torch.max(output[head][0, anchor, ..., 5+index]) )
            id_a, id_b, id_c, id_d, id_e = indexes[0]

        one_hot = torch.zeros((1, *output[head].size()), device=self.device, dtype=torch.float32)
        one_hot[id_a, id_b, id_c, id_d, id_e] = 1
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot * output[head])

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]

        target = features[-1]
        target = target[0, :]

        weights = torch.mean(grads_val, axis=(2, 3))[0, :]
        cam = torch.zeros(target.shape[1:], device=self.device, dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = cam = torch.where(cam > 0, cam, torch.tensor(0., device=self.device))
        resize = Transforms.Compose([ Transforms.ToPILImage(), Transforms.Resize(input.shape[2:]), Transforms.ToTensor() ])
        cam = resize(torch.stack([cam.cpu()]))[0] # torch resizes only 3D or moreD tensors, not 2D
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam.cpu().numpy()
