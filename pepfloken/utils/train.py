import wandb

import numpy as np
import torch
from easydict import EasyDict

from pepfloken.utils.misc import BlackHole



def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    elif cfg.type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)
def get_optimizer2(cfg, model):


    def cfg_get(cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    base_lr = cfg_get(cfg, "lr")
    weight_decay = cfg_get(cfg, "weight_decay", 0.0)
    beta1 = cfg_get(cfg, "beta1", 0.9)
    beta2 = cfg_get(cfg, "beta2", 0.999)

    node_edge_lr_mult = cfg_get(cfg, "node_edge_lr_mult", 0.1)
    sidechain_lr_mult = cfg_get(cfg, "sidechain_lr_mult", 1.0)
    other_lr_mult = cfg_get(cfg, "other_lr_mult", 1.0)

    node_edge_params = []
    sidechain_params = []
    other_params = []

    node_edge_names = []
    sidechain_names = []
    other_names = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "node_embedder" in name or "edge_embedder" in name:
            node_edge_params.append(p)
            node_edge_names.append(name)

        elif "sidechain_decoder" in name:
            sidechain_params.append(p)
            sidechain_names.append(name)

        else:
            other_params.append(p)
            other_names.append(name)

    param_groups = []

    if node_edge_params:
        param_groups.append({
            "params": node_edge_params,
            "lr": base_lr * node_edge_lr_mult,
            "weight_decay": weight_decay,
            "name": "node_edge_embedder",
        })

    if sidechain_params:
        param_groups.append({
            "params": sidechain_params,
            "lr": base_lr * sidechain_lr_mult,
            "weight_decay": weight_decay,
            "name": "sidechain_decoder",
        })

    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": base_lr * other_lr_mult,
            "weight_decay": weight_decay,
            "name": "other",
        })

    opt_type = cfg_get(cfg, "type")

    if opt_type == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            betas=(beta1, beta2),
        )

    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(beta1, beta2),
        )

    else:
        raise NotImplementedError("Optimizer not supported: %s" % opt_type)

    if hasattr(model, "module"):
        is_main = True
    else:
        is_main = True

    if is_main:
        print("[Optimizer param groups]")
        print(f"  node_edge_embedder: {len(node_edge_params)} tensors, lr={base_lr * node_edge_lr_mult:g}")
        print(f"  sidechain_decoder:  {len(sidechain_params)} tensors, lr={base_lr * sidechain_lr_mult:g}")
        print(f"  other:              {len(other_params)} tensors, lr={base_lr * other_lr_mult:g}")

    return optimizer

def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    elif cfg.type is None:
        return BlackHole()
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def get_warmup_sched(cfg, optimizer):
    if cfg is None: return BlackHole()
    lambdas = [lambda it : (it / cfg.max_iters) if it <= cfg.max_iters else 1 for _ in optimizer.param_groups]
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return warmup_sched


def log_losses(loss, loss_dict, scalar_dict, it, tag, logger=BlackHole(), writer=BlackHole()):
    logstr = '[%s] Iter %05d' % (tag, it)
    logstr += ' | loss %.4f' % loss.item()
    for k, v in loss_dict.items():
        if type(v) == torch.Tensor:
            logstr += ' | loss(%s) %.4f' % (k, v.item())
        else:
            logstr += ' | loss(%s) %.4f' % (k, v)
    for k, v in scalar_dict.items():
        logstr += ' | %s %.4f' % (k, v.item() if isinstance(v, torch.Tensor) else v)
    logger.info(logstr)




class ScalarMetricAccumulator(object):

    def __init__(self):
        super().__init__()
        self.accum_dict = {}
        self.count_dict = {}

    @torch.no_grad()
    def add(self, name, value, batchsize=None, mode=None):
        assert mode is None or mode in ('mean', 'sum')

        if mode is None:
            delta = value.sum()
            count = value.size(0)
        elif mode == 'mean':
            delta = value * batchsize
            count = batchsize
        elif mode == 'sum':
            delta = value
            count = batchsize
        delta = delta.item() if isinstance(delta, torch.Tensor) else delta

        if name not in self.accum_dict:
            self.accum_dict[name] = 0
            self.count_dict[name] = 0
        self.accum_dict[name] += delta
        self.count_dict[name] += count

    def log(self, it, tag, logger=BlackHole(), writer=BlackHole()):
        summary = {k: self.accum_dict[k] / self.count_dict[k] for k in self.accum_dict}
        logstr = '[%s] Iter %05d' % (tag, it)
        for k, v in summary.items():
            logstr += ' | %s %.4f' % (k, v)
            writer.add_scalar('%s/%s' % (tag, k), v, it)
            wandb.log({f'{tag}/{k}': v}, step=it)
        logger.info(logstr)

    def get_average(self, name):
        return self.accum_dict[name] / self.count_dict[name]


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def sum_weighted_losses(losses, weights):

    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
