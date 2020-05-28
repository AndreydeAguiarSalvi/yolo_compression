import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from collections import OrderedDict
from utils.pruning import create_mask_LTH, apply_mask_LTH

##############
# IMP in cpu #
##############
def IMP_GLOBAL(model, mask, percentage_of_pruning): # Implements Lottery Tickets Hypothesis globally
    valid_values = torch.Tensor(0).cuda()
    for name, param in model.named_parameters():
        if 'bias' not in name and 'bn' not in name and 'BatchNorm' not in name:
            name_ = name.replace('.', '-') # ParameterDict and ModuleDict does not allows '.' as key
            # Getting all the available values to possibly be pruned
            valid_values = torch.cat( ( valid_values, torch.masked_select(param, mask[name_].data.byte()) ) )
    
    # Globally number of neurons to be prunned
    n_pruned_neurons = math.floor(valid_values.shape[0] * percentage_of_pruning)
    # Getting the values to be prunned.
    smallest_values = torch.topk(input=torch.abs(valid_values), k=n_pruned_neurons, largest=False)
    # Getting the higher valid value to be prune.
    # All non-zero elements smaller than higher_of_smallest
    # will be pruned.
    higher_of_smallest = smallest_values.values[-1]

    for name, param in model.named_parameters():
        if 'bias' not in name and 'bn' not in name and 'BatchNorm' not in name:
            name_ = name.replace('.', '-') # ParameterDict and ModuleDict does not allows '.' as key
            # Create the new mask
            with torch.no_grad():
                mask[name_] = torch.nn.Parameter( 
                    torch.where(torch.abs(param) <= higher_of_smallest, torch.tensor(.0), mask[name_]) 
                )

x = torch.ones([5, 3, 416, 416])

###################################
# Convolution with sparse matrix  #
# just possible wheter to_dense() #
###################################
# conv = nn.Conv2d(3, 1, 3)
# data = torch.FloatTensor([-0.85, 0.57, 0.33, 0.97])
# index = torch.LongTensor([ [0, 0, 0, 0], [0, 1, 0, 2], [0, 2, 0, 1], [0, 1, 2, 2] ]) # in_channels, out_channels, x, y
# w = torch.sparse.FloatTensor(index, data, torch.Size([1, 3, 3, 3]))

# y1 = conv(x)
# y2 = F.conv2d(x, w.to_dense())
# print(f"y1: {y1.shape}  y2: {y2.shape}")

class SparseConv(nn.Module):

    def __init__(self, original_conv):
        super(SparseConv, self).__init__()
        
        self.find_non_null_filters(original_conv)
        dict = self.create_splited_convs(original_conv)
        self.fractional_convs = nn.ModuleDict(dict)
    

    def forward(self, x):
        y = []
        # y2 = []
        for (key, value) in self.fractional_convs.items():
            y.append(value(x))
        # for i in reversed(y): y2.append(i)

        return torch.cat(y, dim=1)
        

    def find_non_null_filters(self, conv): # conv.shape is out_channels, in_channels, x, y
        onehot_parameters = torch.sum(torch.abs(conv.weight), dim=(1, 2, 3))
        self.convs_list = torch.where( onehot_parameters > 0, torch.tensor(1), torch.tensor(0) )
        

    def create_splited_convs(self, original_conv):
        sequential_ones = []
        sequential_zeros = []
        result = OrderedDict()
        count_ones = 1
        count_zeros = 1

        for i in range(self.convs_list.shape[0]):
            if self.convs_list[i] == torch.tensor(1): 
                if len(sequential_zeros ) > 0:
                    new_conv = ZeroConv(sequential_zeros, original_conv.kernel_size, original_conv.padding, original_conv.stride)
                    result['zero' + str(count_zeros)] = new_conv
                    count_zeros += 1
                    sequential_zeros = []

                sequential_ones.append(i)
            else:
                if len(sequential_ones) > 0:
                    new_conv = self.create_miniconv_from(original_conv, sequential_ones)
                    result['conv' + str(count_ones)] = new_conv
                    count_ones += 1
                    sequential_ones = []
                
                sequential_zeros.append(1)
        
        if len(sequential_ones) > 0:
            new_conv = self.create_miniconv_from(original_conv, sequential_ones)
            result['conv' + str(count_ones)] = new_conv
        elif len(sequential_zeros) > 0:
            new_conv = ZeroConv(sequential_zeros)
            result['zero' + str(count_zeros)] = new_conv
        
        return result


    def create_miniconv_from(self, original_conv, channels_list):
        new_conv = nn.Conv2d(
            in_channels=original_conv.in_channels, out_channels=len(channels_list), 
            kernel_size=original_conv.kernel_size, padding=original_conv.padding,
            stride=original_conv.stride, groups=original_conv.groups,
            bias = True if original_conv.bias is not None else False
        )
        new_conv.weight.data = original_conv.weight[ channels_list[0] : channels_list[-1]+1 ]
        if original_conv.bias is not None:
            new_conv.bias.data = original_conv.bias[ channels_list[0] : channels_list[-1]+1 ]

        return new_conv


class ZeroConv(nn.Module):

    def __init__(self, channels_list, kernel_size, padding, stride):
        super().__init__()    
        self.channels = len(channels_list)
        self.kernel = kernel_size
        self.padding = padding
        self.stride = stride
    
    
    def forward(self, input):
        batch_size = input.shape[0]
        width, height = input.shape[-2:]
        width = self.compute_size(width, self.kernel[0], self.padding[0], self.stride[0])
        height = self.compute_size(height, self.kernel[1], self.padding[1], self.stride[1])
        
        return torch.zeros(torch.Size([batch_size, self.channels, width, height]))
    

    def compute_size(self, dimension, kernel, padding, stride):
        return int( ( (dimension - kernel + 2 * padding) / stride) + 1)

# conv1 = nn.Conv2d(3, 8, 3)

# conv2 = deepcopy(conv1)
# mask = create_mask_LTH(conv2)
# IMP_GLOBAL(conv2, mask, .5)
# apply_mask_LTH(conv2, mask)

# conv3 = SparseConv(conv2)

conv1 = nn.Conv2d(3, 8, 3)
conv1.weight[2] = torch.zeros(torch.Size([3, 3, 3]))
conv1.bias[2] = torch.tensor(0)
conv1.weight[5] = torch.zeros(torch.Size([3, 3, 3]))
conv1.bias[5] = torch.tensor(0)
conv1.weight[6] = torch.zeros(torch.Size([3, 3, 3]))
conv1.bias[6] = torch.tensor(0)
print(f"Conv1: {conv1}")
print(f"Usefull filters: {torch.where( torch.sum(torch.abs(conv1.weight), dim=(1, 2, 3)) > 0, torch.tensor(1), torch.tensor(0) )}")

conv2 = SparseConv(conv1)
print(f"Conv2: {conv2}")

y1 = conv1(x)
y2 = conv2(x)

print(y1.shape)
print(y2.shape)
print(torch.mean( torch.abs(y1.data - y2.data), dim=torch.Size([2, 3]) ) )
# print( torch.abs(y1.data - y2.data) )

# for i in conv1.named_parameters(): print(i)
# print()
# for i in conv2.named_parameters(): print(i)