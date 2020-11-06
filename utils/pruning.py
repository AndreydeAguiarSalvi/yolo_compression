import math
import torch
import numpy as np
import torch.nn as nn
from utils.layers import SynFlowLinear, SynFlowConv2d, SynFlowBatchNorm1d, SynFlowBatchNorm2d, SynFlowIdentity1d, SynFlowIdentity2d

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
    nc = (model.module_defs[-1]['classes'] + 5) * 3
    for name, param in model.named_parameters():
        if 'PEP' not in name and param.shape[0] == nc: pass # not create mask of last layer
        elif 'bias' not in name and 'bn' not in name and 'BatchNorm' not in name:
            name_ = name.replace('.', '-') # ParameterDict and ModuleDict does not allows '.' as key
            mask[name_] = nn.Parameter( torch.ones_like(param), requires_grad = False )

    return nn.ParameterDict(mask)


def IMP_LOCAL(model, mask, percentage_of_pruning): # Implements Lottery Tickets Hypothesis locally
    nc = (model.module_defs[-1]['classes'] + 5) * 3
    for name, param in model.named_parameters():
        conv_before_yolo = 'PEP' not in name and param.shape[0] == nc
        if not conv_before_yolo and 'bias' not in name and 'bn' not in name and 'BatchNorm' not in name:
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
    nc = (model.module_defs[-1]['classes'] + 5) * 3
    for name, param in model.named_parameters():
        conv_before_yolo = 'PEP' not in name and param.shape[0] == nc
        if not conv_before_yolo and 'bias' not in name and 'bn' not in name and 'BatchNorm' not in name:
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
        conv_before_yolo = 'PEP' not in name and param.shape[0] == nc
        if not conv_before_yolo and 'bias' not in name and 'bn' not in name and 'BatchNorm' not in name:
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


#################
# Synaptic Flow #
#################
class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf 
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
    
    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0 
        for mask, _ in self.masked_parameters:
             remaining_params += mask.detach().cpu().numpy().sum()
             total_params += mask.numel()
        return remaining_params, total_params


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
      
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)

        imgs, _, _, _ = next(iter(dataloader))
        input_dim = list(imgs[0].shape)
        input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()
        
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)

def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf


def prunable(module, batchnorm, residual):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (SynFlowLinear, SynFlowConv2d))
    if batchnorm:
        isprunable |= isinstance(module, (SynFlowBatchNorm1d, SynFlowBatchNorm2d))
    if residual:
        isprunable |= isinstance(module, (SynFlowIdentity1d, SynFlowIdentity2d))
    return isprunable


def prune_loop(model, loss, pruner, dataloader, device, sparsity, 
               schedule, scope, epochs, reinitialize=False, train_mode=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        pruner.mask(sparse, scope)
    
    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    if np.abs(remaining_params - total_params*sparsity) >= 5:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
        quit()


def masked_parameters(model, bias=False, batchnorm=False, residual=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            if param is not module.bias or bias is True:
                yield mask, param