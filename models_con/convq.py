import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
from tqdm.auto import tqdm
import functools
from torch.utils.data import DataLoader
import os
import argparse

import pandas as pd

from models_con.edge import EdgeEmbedder
from models_con.node import NodeEmbedder
from pepfloken.modules.common.layers import sample_from, clampped_one_hot
from pepfloken.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms
from pepfloken.modules.common.geometry import construct_3d_basis
from pepfloken.utils.data import mask_select_data, find_longest_true_segment, PaddingCollate
from pepfloken.utils.misc import seed_all
from pepfloken.utils.train import sum_weighted_losses
from torch.nn.utils import clip_grad_norm_

from pepfloken.modules.so3.dist import centered_gaussian,uniform_so3
from pepfloken.modules.common.geometry import batch_align, align

from tqdm import tqdm

import wandb

from pepfloken.data import so3_utils
from pepfloken.data import all_atom

from models_con.pep_dataloader import PepDataset

from pepfloken.utils.misc import load_config
from pepfloken.utils.train import recursive_to
from easydict import EasyDict

from models_con.utils import process_dic
from models_con.torsion import get_torsion_angle, torsions_mask
import models_con.torus as torus

import gc
from pepfloken.modules.common.geometry import  get_backbone_dihedral_angles

from copy import deepcopy
from pepfloken.utils.data import PaddingCollate
collate_fn = PaddingCollate(eight=False)
from pepfloken.utils.train import recursive_to
def angle_to_sin_cos(degrees,isdegree=True):
    shape = [i for i in degrees.shape]
    shape[-1] *= 2
    degrees = degrees.unsqueeze(-1)
    if isdegree:
        radians = torch.deg2rad(degrees).unsqueeze(-1)
    else:
        radians = degrees.unsqueeze(-1)
    return torch.concat([torch.cos(radians.clone()),torch.sin(radians.clone())],dim=-1).reshape(shape)
resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}
class Convq(nn.Module):
    def __init__(self,cfg,vae_model):
        super().__init__()
        self._model_cfg = cfg.encoder
        self._interpolant_cfg = cfg.interpolant
        self.nodelinear = nn.Linear(128,256)
        self.edge_linear = nn.Linear(64,256)
        self.node_embedder = NodeEmbedder(cfg.encoder.node_embed_size,max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(cfg.encoder.edge_embed_size,max_num_heavyatoms)
        self.sample_structure = self._interpolant_cfg.sample_structure
        self.sample_sequence = self._interpolant_cfg.sample_sequence
        self.vae_model = vae_model
        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
        self.vq_size = 1024 
        self.psi_transform = nn.Linear(2,256)
    def encode(self, batch):
        rotmats_1 =  construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N] )
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        seqs_1 = batch['aa']

        angles_1 = batch['torsion_angle'].clone()
        angles_1[:,:,1:]=angles_1[:,:,1:]*~batch['generate_mask'].unsqueeze(-1)
        angle_mask = batch['torsion_angle_mask']
        angle_mask[:,:,1:]=angle_mask[:,:,1:]*~batch['generate_mask'].unsqueeze(-1)
        vq_emb = batch['vq_emb']

        context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
        structure_mask = context_mask if self.sample_structure else None
        sequence_mask = context_mask if self.sample_sequence else None
        node_embed = self.node_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        edge_embed = self.edge_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)

        return rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed,vq_emb

    def zero_center_part(self,pos,gen_mask,res_mask):

        center = torch.sum(pos * gen_mask[...,None], dim=1) / (torch.sum(gen_mask,dim=-1,keepdim=True) + 1e-8) 
        center = center.unsqueeze(1) 
        pos = pos - center
        pos = pos * res_mask[...,None]
        return pos,center

    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k 

    def forward(self, batch):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask,angle_mask= batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed,vq_emb = self.encode(batch) 
        vq_emb = self.vae_model.vq_layer.embeddings(vq_emb)
        node_emb = self.nodelinear(node_embed)
        edge_emb = self.edge_linear(edge_embed.mean(-2))
        x_recon = self.vae_model.decoder(vq_emb.squeeze()+node_emb+edge_emb+self.psi_transform(angle_to_sin_cos(angles_1[:,:,:1])),res_mask.to(torch.bool))
        x_seq = self.vae_model.seqhead(x_recon)
        x_ang = self.vae_model.anghead(x_recon)
        t_ang = batch['torsion_angle'].clone()[:,:,1:]
        angle_mask = t_ang[gen_mask!=0]!=0
        angle_mask = torch.concat([angle_mask.unsqueeze(-1),angle_mask.unsqueeze(-1)],dim=-1).reshape(-1,8)
        t_ang = angle_to_sin_cos(t_ang,isdegree=False).to(x_ang.device).reshape(num_batch,num_res,-1)
        aa_loss = F.cross_entropy(x_seq[gen_mask!=0],batch['aa'][gen_mask!=0])
        vq_loss =F.mse_loss(x_ang[gen_mask!=0][angle_mask],t_ang[gen_mask!=0][angle_mask])
        return {
            'aa_loss': aa_loss,
            'vq_loss': vq_loss,
        }
    @torch.no_grad()
    def decode(self, batch,angles_1,vq_emb):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask,angle_mask= batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()
        rotmats_1, trans_1, _, seqs_1, node_embed, edge_embed,_ = self.encode(batch) 
        vq_emb = self.vae_model.vq_layer.embeddings(vq_emb)
        node_emb = self.nodelinear(node_embed)
        edge_emb = self.edge_linear(edge_embed.mean(-2))
        x_recon = self.vae_model.decoder(vq_emb.squeeze()+node_emb+edge_emb+self.psi_transform(angle_to_sin_cos(angles_1[:,:,:1])).squeeze(),res_mask.to(torch.bool))
        x_seq = self.vae_model.seqhead(x_recon)
        x_ang = self.vae_model.anghead(x_recon)


        return x_seq, x_ang
