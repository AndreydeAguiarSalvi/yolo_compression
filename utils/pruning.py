import math
import torch
import torch.nn as nn


####################
# Generall Methods #
####################
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


##############################
# Lottery Tickets Hypothesis #
##############################
def apply_mask_LTH(model, mask):
    with torch.no_grad():
        for name, param in mask.items():
            name_ = name.replace('-', '.') # Changing to the original key
            model.state_dict()[name_].data.copy_( model.state_dict()[name_].data.mul(param.data) )


def create_mask_LTH(model): # Create mask as Lottery Tickets Hypothesis
    from collections import OrderedDict
    mask = OrderedDict()
    for name, param in model.named_parameters():
        if 'bias' not in name and 'bn' not in name and 'BatchNorm' not in name:
            name_ = name.replace('.', '-') # ParameterDict and ModuleDict does not allows '.' as key
            mask[name_] = nn.Parameter( torch.ones_like(param), requires_grad = False )

    return nn.ParameterDict(mask)


def IMP_LOCAL(model, mask, percentage_of_pruning): # Implements Lottery Tickets Hypothesis locally
    for name, param in model.named_parameters():
        if 'bias' not in name and 'bn' not in name and 'BatchNorm' not in name:
            name_ = name.replace('.', '-') # ParameterDict and ModuleDict does not allows '.' as key
            # Locally number of neurons to be prunned
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
                    torch.where(torch.abs(param) <= higher_of_smallest, torch.tensor(.0, device='cuda'), mask[name_]) 
                )


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
                    torch.where(torch.abs(param) <= higher_of_smallest, torch.tensor(.0, device='cuda'), mask[name_]) 
                )


#############################
# Continuous Sparsification #
#############################
def apply_mask_CS(model, mask):
    for name, param in mask.items():
        name_ = name.replace('-', '.') # Changing to the original key
        model.state_dict()[name_] = model.state_dict()[name_] * param


def create_mask_CS(model, init_value = .0): # Create mask as Continuous Sparsification
    from collections import OrderedDict
    mask = OrderedDict()
    for name, param in model.named_parameters():
        if 'bias' not in name and 'bn' not in name and 'BatchNorm' not in name:
            name_ = name.replace('.', '-') # ParameterDict and ModuleDict does not allows '.' as key
            mask[name_] = nn.Parameter( param.new_full(param.shape, fill_value = init_value), requires_grad = True ) 

    return nn.ParameterDict(mask)


def CS(mask, initial_value, Beta): # Implements prune() from Continuous Sparcification
    for key in mask.keys():
        mask[key].data = torch.clamp(Beta * mask[key], max = initial_value)


def compute_mask(mask, mask_initial_value, Beta = 1., is_ticket = False): # Implements from apply_mask() from CS
    from collections import OrderedDict
    pseudo_mask = OrderedDict()
    scaling = 1. / torch.sigmoid( torch.tensor(mask_initial_value) )
    for key in mask.keys():
        if is_ticket: pseudo_mask[key] = nn.Parameter( (mask[key] > 0).float() )
        else: pseudo_mask[key] = nn.Parameter( torch.sigmoid(Beta * mask[key]) )
        pseudo_mask[key] = nn.Parameter( scaling * pseudo_mask[key] )
    
    return nn.ParameterDict(pseudo_mask)


def compute_masked_weights(model, mask): # Implements line from layers.py/SoftMaskedConv2.forward
    from collections import OrderedDict
    masked_weights = OrderedDict()
    for name, param in mask.named_parameters():
        name_ = name.replace('-', '.')
        masked_weights[name] = model[name_] * mask[name]
    
    return nn.ParameterDict(masked_weights)