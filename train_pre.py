import wandb

import os
import shutil
import argparse
import torch
import torch.cuda.amp as amp
import torch.distributed as distrib
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from torch.utils.data.distributed import DistributedSampler
from pepfloken.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from pepfloken.utils.data import PaddingCollate
from pepfloken.utils.train import ScalarMetricAccumulator, count_parameters, get_optimizer, get_scheduler, log_losses, recursive_to, sum_weighted_losses
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models_con.pep_dataloader import PepDataset
from pepfloken.modules.protein.constants import AA
from models_con.pepfloken import PepFloken
import math
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_cosine_scheduler(
    optimizer,
    warmup_steps,
    total_steps,
    min_lr_ratio=0.05,
):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
def process_ddp_state_dict(state_dict, is_ddp_model):
    new_state_dict = {}
    if is_ddp_model:
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                new_state_dict[f'module.{k}'] = v
            else:
                new_state_dict[k] = v
    else:
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
    return new_state_dict
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/pretrain.yaml')
    parser.add_argument('--logdir', type=str, default="./logs")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default='result')
    parser.add_argument('--name', type=str, default='PepFloken_pre')
    parser.add_argument('--save_dir', type=str, default='test')
    parser.add_argument('--local_rank', type=int, help='Local rank. Necessary for using the torch.distributed.launch utility.')
    parser.add_argument('--continuepath', type=str, default=None)
    args = parser.parse_args()
    local_rank = os.environ.get('LOCAL_RANK', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', -1))

    local_rank = int(local_rank) 
    torch.cuda.set_device(local_rank)
    vq_tokendic = torch.load('vaemodel2_codebook.bin')
    token_to_vq = {}
    aadic = {
        'A': 0,   
        'C': 1,   
        'D': 2,   
        'E': 3,   
        'F': 4,   
        'G': 5,   
        'H': 6,   
        'I': 7,   
        'K': 8,   
        'L': 9,   
        'M': 10,  
        'N': 11,  
        'P': 12,  
        'Q': 13,  
        'R': 14,  
        'S': 15,  
        'T': 16,  
        'V': 17,  
        'W': 18,  
        'Y': 19,  
        'X': 20,   
        'b': 21   
    }
    for k in vq_tokendic.keys():
        for v in vq_tokendic[k]:
            token_to_vq[v] = k
    vq_seqtokendic={}
    for k,v in vq_tokendic.items():
        vq_seqtokendic[aadic[k]] = v
        
    inddic = {}
    for k,v in token_to_vq.items():
        inddic[k] = aadic[v]
    vq_to_seq_matrix = torch.zeros((1024,22))
    for k,v in inddic.items():
        vq_to_seq_matrix[k,v] = 1
    seq_to_vq_matrix = vq_to_seq_matrix.T


    config, config_name = load_config(args.config)
    seed_all(config.train.seed+local_rank*100)

    config.model['vq_tokendic'] = vq_tokendic
    config.model['vq_to_seq_matrix'] = vq_to_seq_matrix
    config.model['seq_to_vq_matrix'] = seq_to_vq_matrix
    writer = BlackHole()

    log_dir = os.path.join(os.path.curdir,args.resume)
    ckpt_dir = os.path.join(log_dir, args.name)
    if not os.path.exists(ckpt_dir):
        
        os.mkdir(ckpt_dir)
    logger = get_logger('train', None, local_rank)
    logger.info(args)
    logger.info(config)
    distrib.init_process_group(backend="nccl")
    logger.info('Loading datasets...')
    train_dataset = PepDataset(structure_dir = config.dataset.train.structure_dir, dataset_dir = config.dataset.train.dataset_dir,
                                            name = config.dataset.train.name, transform=None, reset=config.dataset.train.reset)
    val_dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
                                            name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=args.num_workers, pin_memory=True)
    train_iterator = inf_iterator(train_loader)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(train_dataset)))

    logger.info('Building model...')
    model = DDP(PepFloken(config.model).to(local_rank), device_ids=[local_rank])

    if args.continuepath is not None:
        logger.info('Resuming from checkpoint: %s' % args.continuepath)
        ckpt = torch.load(
            args.continuepath, 
            map_location=f'cuda:{local_rank}',
            weights_only=False
        )
        model_state_dict = process_ddp_state_dict(ckpt['model'], is_ddp_model=True)
        model.load_state_dict(model_state_dict, strict=False)

        
    logger.info('Number of parameters: %d' % count_parameters(model))

    optimizer = get_optimizer(config.train.optimizer, model)
    if config.train.scheduler.type=='warmup_cosine':
        scheduler = get_warmup_cosine_scheduler(     optimizer,config.train.scheduler.warmup_steps,config.train.scheduler.total_steps)
    else:
        scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    def train(it):
        time_start = current_milli_time()
        model.train()

        batch = recursive_to(next(train_iterator), local_rank)

        loss_dict,loss_dict2 = model(batch) 
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        time_forward_end = current_milli_time()


        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    param.grad[torch.isnan(param.grad)] = 0

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        optimizer.step()
        if config.train.scheduler.type=='warmup_cosine':
            scheduler.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        if local_rank == 0:
            scalar_dict = {}
            scalar_dict.update({
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
            })
            log_losses(loss, loss_dict2, scalar_dict, it=it, tag='train', logger=logger)
            del loss
    def validate(it):
        scalar_accum = ScalarMetricAccumulator()
        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)

                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                scalar_accum.add(name='loss', value=loss, batchsize=len(batch['aa']), mode='mean')
            
        avg_loss = scalar_accum.get_average('loss')
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss
    best_loss  = None
    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 and local_rank == 0:
                    
                ckpt_path = os.path.join(ckpt_dir, 'Pepfloken_pre.pt')
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
        distrib.destroy_process_group()