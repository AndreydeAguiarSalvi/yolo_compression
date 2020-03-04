import torch
import torch.nn as nn
from models import YOLO_Teacher
from utils.pruning import sum_of_the_weights, create_mask

class MultiBiasConv(nn.Module):

    def __init__(self, in_channels, out_channels, n_bias, kernel_size=(3, 3), stride=1, pad=0):
        super(MultiBiasConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=int(in_channels), out_channels=int(out_channels/n_bias), bias=False, kernel_size=kernel_size, padding=pad, stride=stride)
        self.bias = torch.nn.Parameter( torch.Tensor(n_bias), requires_grad=True )


    def forward(self, x):
        x = self.conv(x)
        y = []
        for b in self.bias:
            y.append(x + b)
        
        return torch.cat(y, dim=1)
    

class Reduced3x3Conv(nn.Module):
    
    def __init__(self, input_size, output_size, n_bias):
        super(Reduced3x3Conv, self).__init__()
        self.conv1 = MultiBiasConv(input_size, output_size, n_bias, (3, 1))
        self.conv2 = MultiBiasConv(output_size, output_size, n_bias, (1, 3))
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class FullyResidual(nn.Module):
    
    """Fully Residual Convolutional Neural Network"""
    def __init__(self, input_size, output_size):
        super(FullyResidual, self).__init__()
        # Input = img
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=int(output_size/4), bias=True, kernel_size=3, padding=1, stride=1)
        # Input = conv1
        self.conv2 = MultiBiasConv(in_channels=int(output_size/4), out_channels=int(output_size/4), n_bias=4)
        # Input = conv1 + conv2
        self.conv3 = MultiBiasConv(in_channels=int(output_size/2), out_channels=int(output_size/4), n_bias=4)
        # Input = conv1 + conv2 + conv3
        self.conv4 = MultiBiasConv(in_channels=int(output_size*3)/4, out_channels=int(output_size/4), n_bias=4)
        # Input = conv1 + conv2 + conv3 + conv4
        self.conv5 = nn.Conv2d(in_channels=int(output_size), out_channels=int(output_size-1), bias=True, kernel_size=3, padding=0, stride=3)

        self.F = torch.nn.functional.leaky_relu


    def forward(self, x):
        x1 = self.F( self.conv1(x) )
        x2 = self.F( self.conv2(x1) )
        x3 = self.F( self.conv3(torch.cat([x1, x2], dim=1)) )
        x4 = self.F( self.conv4(torch.cat([x1, x2, x3], dim=1)) )
        x5 = self.F( self.conv5(torch.cat([x1, x2, x3, x4], dim=1)) )
        
        return x5


class MultiFullyResidualMultiBiasedBlockNetwork(nn.Module):

    """Fully Residual Convolutional Neural Network"""
    def __init__(self, out_size):
        super(MultiFullyResidualMultiBiasedBlockNetwork, self).__init__()
        self.block1 = FullyResidual(input_size=3, output_size=out_size)
        self.block2 = FullyResidual(input_size=out_size-1, output_size=out_size)
        self.block3 = FullyResidual(input_size=out_size-1, output_size=out_size)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x


# teacher = YOLO_Teacher(cfg='cfg/yolov3.cfg')
# print(f'Weights from teacher: { sum_of_the_weights(create_mask(teacher)) }')
# student = MultiFullyResidualMultiBiasedBlockNetwork(out_size=256)
# print(f'Weights from student: {sum_of_the_weights(create_mask(student))}')

# teacher.eval()
# student.train()

# x = torch.Tensor(18, 3, 416, 416)

# fts = teacher(x)
# for ft in fts:
#     if isinstance(ft, tuple):
#         for t in ft:
#             print(t.shape)
#     else:    
#         print(ft.shape)

# output = student(x)
# print(f'Student: {output.shape}')



# net1 = nn.Sequential(nn.Conv2d(3, 32, 3), nn.Conv2d(32, 64, 3), nn.Conv2d(64, 128, 3))
# net2 = nn.Sequential(
#     # First
#     MultiBiasConv(3, 32, 16, (3, 3)) ,
#     # Second
#     MultiBiasConv(32, 64, 16, (3, 3)),
#     # Third
#     MultiBiasConv(64, 128, 16, (3, 3)),
# )
# net3 = nn.Sequential(
#     Reduced3x3Conv(3, 32, 16),
#     Reduced3x3Conv(32, 64, 16),
#     Reduced3x3Conv(64, 128, 16)
# )

# m1 = create_mask(net1)
# m2 = create_mask(net2)
# m3 = create_mask(net3)
# print(f'net1: {sum_of_the_weights(m1)}')
# print(f'net2: {sum_of_the_weights(m2)}')
# print(f'net3: {sum_of_the_weights(m3)}')

# x = torch.Tensor(16, 3, 416, 416)
# y1 = net1(x)
# y2 = net2(x)
# y3 = net3(x)

# print(f'y1: {y1.shape}')
# print(f'y2: {y2.shape}')
# print(f'y3: {y3.shape}')

# for name, param in net2.named_parameters():
#     print(f'model param: {name}')

# for name, param in m2.named_parameters():
#     print(f'mask param: {name}')

c1 = nn.Conv2d(3, 32, 1)
c2 = MultiBiasConv(3, 32, 16, 1)
x = torch.Tensor(18, 3, 416, 416)
print(f'Y1: {c1(x).shape}')
print(f'Y2: {c2(x).shape}')