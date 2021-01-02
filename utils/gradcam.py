# Based on https://github.com/jacobgil/pytorch-grad-cam
# The original code does't work with YOLO
# This is an adaptation from https://github.com/jacobgil/pytorch-grad-cam to work with YOLO
# The original code assumes that the models have two functions: features and classifier
# Paper: https://arxiv.org/pdf/1610.02391v1.pdf

import cv2
import torch
import numpy as np
from torch.autograd import Function
import torchvision.transforms as Transforms
from utils.utils import compute_loss

relu6 = torch.nn.ReLU6(inplace=True)

def show_cam_on_image(img, mask, path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(path, np.uint8(255 * cam))


def deprocess_image(img, normalization=True):
    if normalization:
        img = img - np.mean(img)
        img = img / (np.std(img) + 1e-5)
        img = img * 0.1
        img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


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
            indexes = torch.nonzero( output[head] == torch.max(output[head][0, anchor, ..., int(5+index)]) )
            id_a, id_b, id_c, id_d, id_e = indexes[0]

        one_hot = torch.zeros((*output[head].size()), device=self.device, dtype=torch.float32)
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


def generic_backward(conv_output, grad_output):
    positive_mask_1 = (conv_output > 0).type_as(grad_output)
    positive_mask_2 = (grad_output > 0).type_as(grad_output)
    grad_input = torch.addcmul(torch.zeros(conv_output.size()).type_as(conv_output),
                                torch.addcmul(torch.zeros(conv_output.size()).type_as(conv_output), grad_output,
                                                positive_mask_1), positive_mask_2)
    return grad_input

class GuidedBackpropLeakyReLU(Function):
    @staticmethod
    def forward(self, input_img):
        output = torch.where(input_img > 0, input_img, 0.1*input_img)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        conv_output, fcn_output = self.saved_tensors

        return generic_backward(conv_output, grad_output)


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        conv_output, fcn_output = self.saved_tensors

        return generic_backward(conv_output, grad_output)


class GuidedBackpropReLU6(Function):
    @staticmethod
    def forward(self, input_img):
        device = input_img.device
        zero = torch.tensor([.0], device=device)
        six = torch.tensor([6.], device=device)
        positive_values = torch.where(input_img > 0, input_img, zero)
        output = torch.where(positive_values < 6, positive_values, six)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        conv_output, fcn_output = self.saved_tensors

        return generic_backward(conv_output, grad_output)


class GuidedBackpropSigmoid(Function):
    @staticmethod
    def forward(self, input_img):
        output = torch.sigmoid(input_img)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        conv_output, fcn_output = self.saved_tensors
        grad_input = None

        positive_mask = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(conv_output.size()).type_as(conv_output), grad_output, positive_mask)
        return grad_input


class GuidedBackpropSwish(Function):
    @staticmethod
    def forward(self, input_img):
        output = input_img * torch.sigmoid(input_img)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        conv_output, fcn_output = self.saved_tensors

        return generic_backward(conv_output, grad_output)


class GuidedBackpropHardSwish(Function):
    @staticmethod
    def forward(self, input_img):
        output = input_img.mul(relu6(input_img+3.)/6.)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        conv_output, fcn_output = self.saved_tensors

        return generic_backward(conv_output, grad_output)


class GuidedBackpropReLUModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(iter(model.parameters())).device

        def recursive_function_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_function_apply(module)
                if module.__class__.__name__ == 'LeakyReLU':
                    module_top._modules[idx] = GuidedBackpropLeakyReLU.apply
                elif module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                elif module.__class__.__name__ == 'ReLU6':
                    module_top._modules[idx] = GuidedBackpropReLU6.apply
                elif module.__class__.__name__ == 'Sigmoid':
                    module_top._modules[idx] = GuidedBackpropSigmoid.apply
                elif module.__class__.__name__ == 'Swish':
                    module_top._modules[idx] = GuidedBackpropSwish.apply
                elif module.__class__.__name__ == 'HardSwish':
                    module_top._modules[idx] = GuidedBackpropHardSwish.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_function_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, head=0, anchor=0, index=None):
        _, output = self.forward(input)

        if index == None:           # [3times [bs, anchors, grid, grid, xywh + classes] ]
            indexes = torch.nonzero( output[head] == torch.max(output[head][0, anchor, ..., 5:]) )
            id_a, id_b, id_c, id_d, id_e = indexes[0]
        else:
            indexes = torch.nonzero( output[head] == torch.max(output[head][0, anchor, ..., int(5+index)]) )
            id_a, id_b, id_c, id_d, id_e = indexes[0]

        one_hot = torch.zeros((*output[head].size()), device=self.device, dtype=torch.float32)
        one_hot[id_a, id_b, id_c, id_d, id_e] = 1
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot * output[head])
        one_hot.backward(retain_graph=True)
        
        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output
