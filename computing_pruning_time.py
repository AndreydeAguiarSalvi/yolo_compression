import os
import torch
import datetime
from models import Darknet
from utils.utils import init_seeds
from utils.torch_utils import select_device
from utils.parse_config import parse_data_cfg
from utils.datasets import LoadImagesAndLabels
from utils.my_utils import create_config, create_prune_argparser
from utils.pruning import sum_of_the_weights, create_backup, rewind_weights, create_mask, apply_mask, IMP_LOCAL


def prune_on_gpu(model, mask, backup, mini_batch, cfg, device):

    for i in range(5): # 5 inferences
        # Apply mask
        mini_batch = mini_batch.to('cpu')
        mask = mask.to(device)
        model, mask = apply_mask(model, mask)
        mask = mask.to('cpu')
        mini_batch = mini_batch.to(device)
        # Perform a prediction
        outputs = model(mini_batch)
        # Pseudo backward

    # Prune and apply mask
    mini_batch = mini_batch.to('cpu')
    mask = mask.to(device)
    IMP_LOCAL(model, mask, cfg['pruning_rate'])
    mask = mask.to('cpu')
    backup = backup.to(device)
    rewind_weights(model, backup)
    backup = backup.to('cpu')
    mask = mask.to(device)
    model, mask = apply_mask(model, mask)
    mask = mask.to(device)

    # Last inference
    outputs = model(mini_batch)


def prune_on_cpu(model, mask, backup, mini_batch, cfg, device):
    
    for i in range(5): # 5 inferences
        # Apply mask
        model.to('cpu')
        model, mask = apply_mask(model, mask)
        model.to(device)
        
        # Performs a prediction
        outputs = model(mini_batch)
        # Pseudo backward

    # Prune and apply mask
    model = model.to('cpu')
    IMP_LOCAL(model, mask, cfg['pruning_rate'])
    rewind_weights(model, backup)
    model, maks = apply_mask(model, mask)
    model = model.to(device)

    # Last inference
    outputs = model(mini_batch)


def main():
    args = create_prune_argparser()
    config = create_config(args)
    
    # Initialize
    init_seeds(seed = 0) 

    model = Darknet(cfg = config['cfg'], arc = config['arc'])
    mask = create_mask(model)
    bckp = create_backup(model)
    device = select_device(config['device'])

    model = model.to(device)
    # print('Making forwards by 100 iterations')
    # mask = mask.to(device)
    # x = torch.Tensor(10, 3, 416, 416).to(device)
    # for i in range(100):
    #     out = model(x)
    # exit()

    data_dict = parse_data_cfg(config['data'])
    train_path = data_dict['train']

    dataset = LoadImagesAndLabels(
        path = train_path, img_size = config['img_size'][0], batch_size=config['batch_size'],
        augment=True, hyp=config['hyp'], 
        cache_images=config['cache_images'],
    )

    # Dataloader
    nw = min([os.cpu_count(), 18 if 18 > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size = 18, num_workers = nw,
        pin_memory = True, collate_fn = dataset.collate_fn
    )

    # torch.cuda.empty_cache()

    imgs, _, _, _ = next(iter(dataloader))
    imgs = imgs.float() / 255.0
    imgs = imgs.to(device)

    start = datetime.datetime.now()
    print(f'Starting to compute the time at {start}')
    for i in range(10):
        prune_on_cpu(model, mask, bckp, imgs, config, device)
    end = datetime.datetime.now()
    print(f'Ending at {end}')
    result = end - start
    print(f'Time of {result}')


if __name__ == "__main__":
    main()