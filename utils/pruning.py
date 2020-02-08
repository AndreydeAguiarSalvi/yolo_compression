import math
import torch
import torch.nn as nn


def create_mask(model):
    from collections import OrderedDict
    mask = OrderedDict()
    for name, param in model.named_parameters():
        if 'bias' not in name and 'bn' not in name:
            name_ = name.replace('.', '-') # ParameterDict and ModuleDict does not allows '.' as key
            mask[name_] = nn.Parameter( torch.ones_like(param), requires_grad = False )

    return nn.ParameterDict(mask)


def apply_mask(model, mask):
    with torch.no_grad():
        for name, param in mask.items():
            name_ = name.replace('-', '.') # Changing to the original key
            model.state_dict()[name_].data.copy_( model.state_dict()[name_].data.mul(param.data) )

    return model, mask


def create_backup(model):
    from copy import deepcopy
    from collections import OrderedDict
    backup = OrderedDict()
    for name in model.state_dict():
        name_ = name.replace('.', '-') # ParameterDict and ModuleDict does not allows '.' as key
        backup[name_] = nn.Parameter( deepcopy(model.state_dict()[name].data), requires_grad = False )
    
    return nn.ParameterDict(backup)


def rewind_weights(model, backup):
    with torch.no_grad():
        for name, param in backup.items():
            name_ = name.replace('-', '.') # Changing to the original key
            model.state_dict()[name_].data.copy_( param )


def sum_of_the_weights(item):
    count = 0
    for name in item.state_dict():
        count += torch.sum(torch.abs(item.state_dict()[name].data))
    
    return count


def IMP_LOCAL(model, mask, percentage_of_pruning): # Implements an Iterative Magnitude Pruning locally
    for name, param in model.named_parameters():
        if 'bias' not in name and 'bn' not in name:
            name_ = name.replace('.', '-') # ParameterDict and ModuleDict does not allows '.' as key
            # Number of neurons to be prunned
            n_pruned_neurons = math.floor(torch.sum(mask[name_]) * percentage_of_pruning)
            # Getting all the available values to possibly be pruned
            valid_values = torch.masked_select(param, mask[name_].data.byte())
            # Getting the values to be prunned.
            smallest_values = torch.topk(input=torch.abs(valid_values), k=n_pruned_neurons, largest=False)
            # Getting the higher valid value to be prune.
            # All non-zero elements smaller than higher_of_smallest
            # will be pruned.
            higher_of_smallest = smallest_values.values[-1]
            # Create the new mask
            with torch.no_grad():
                mask[name_] = torch.nn.Parameter( 
                    torch.where(torch.abs(param) <= higher_of_smallest, torch.tensor(.0), mask[name_]) 
                    # torch.where(torch.abs(param) <= higher_of_smallest, torch.tensor(.0, device='cuda'), mask[name_]) 
                )