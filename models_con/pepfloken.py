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
from models_con.joint_prediction_head import JointPredictionHead
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
from pepfloken.data import utils as du
import gc

from copy import deepcopy
from pepfloken.utils.data import PaddingCollate
collate_fn = PaddingCollate(eight=False)
from pepfloken.modules.common.geometry import reconstruct_backbone


resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}
import torch
from models_con.atoms_decoderv2 import SidechainDecoderV2
from con_code.utils.loss_fn2 import LocalSidechainLossForGenModel
from pepfloken.modules.common.geometry import reconstruct_backbone, reconstruct_backbone_partially, align, batch_align
from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask
from models_con.loss import (
    rot_vec_mul, invert_rot_mat, global_to_local, local_to_global,
    trans_l2_loss, batch_calculate_angles_torch, get_pep_bond_metric,
    pep_bond_loss, rec_loss, time_dependent_vq_loss
)

class PepFloken(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._model_cfg = cfg.encoder
        self._interpolant_cfg = cfg.interpolant
        self.node_embedder = NodeEmbedder(cfg.encoder.node_embed_size,max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(cfg.encoder.edge_embed_size,max_num_heavyatoms)
        self.joint_prediction_head = JointPredictionHead(cfg.encoder.ipa)
        self.register_load_state_dict_pre_hook(self._migrate_joint_prediction_head_keys)
        self.sidechain_decoder = SidechainDecoderV2(hidden_dim=cfg.encoder.ipa.c_hidden,node_dim=cfg.encoder.ipa.c_s,edge_dim=cfg.encoder.ipa.c_z)
        self.sample_structure = self._interpolant_cfg.sample_structure
        self.sample_sequence = self._interpolant_cfg.sample_sequence

        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
        self.vq_size = 1024
        self.vq_tokendic = cfg.vq_tokendic
        self.vq_to_seq_matrix = cfg.vq_to_seq_matrix
        self.seq_to_vq_matrix = cfg.seq_to_vq_matrix
        self.citation = LocalSidechainLossForGenModel(cfg.loss)

    @staticmethod
    def _migrate_joint_prediction_head_keys(module, state_dict, prefix, *args):
        old_prefix = prefix + 'ga_encoder.'
        new_prefix = prefix + 'joint_prediction_head.'
        for key in tuple(state_dict):
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                state_dict.setdefault(new_key, state_dict.pop(key))

    def encode(self, batch):
        rotmats_1 =  construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N] )
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        seqs_1 = batch['aa']

        angles_1 = batch['torsion_angle'].clone()
        angles_1[:,:,1:]=angles_1[:,:,1:]*~batch['generate_mask'].unsqueeze(-1)
        angle_mask = batch['torsion_angle_mask'].clone()
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
    def center_initialize(self,x_global_t,center,res_mask,rot = True):

        x_global_t = x_global_t - center.unsqueeze(1)
        if rot:
            eps = torch.rand(x_global_t.shape[0],1,1,1,device=x_global_t.device)
            rotmats = uniform_so3(x_global_t.shape[0],1,device=x_global_t.device)
            if rotmats.dim() == 4:
                rotmats = rotmats.squeeze(1)
            x_rot = torch.einsum("bij,bnaj->bnai", rotmats, x_global_t)
            x_global_t = torch.where(eps>0.5,x_global_t,x_rot)
            x_global_t = x_global_t * res_mask[...,None,None]
        return x_global_t

    def forward(self, batch):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask,angle_mask= batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()

        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed,vq_emb= self.encode(batch)
        atoms_pos = batch['pos_heavyatom']
        atom_mask = batch['mask_heavyatom']
        trans_1_c,center = self.zero_center_part(trans_1,gen_mask,res_mask)
        trans_atoms_pos = self.center_initialize(atoms_pos ,center,res_mask)
        x_local_1 = global_to_local(trans_atoms_pos,trans_1.unsqueeze(2),rotmats_1.unsqueeze(2))

        vq_emb = vq_emb.reshape(num_batch,-1)

        vq_emb_simplex = clampped_one_hot(vq_emb, self.vq_size).float() * self.k * 2 - self.k
        with torch.no_grad():
            t = torch.rand((num_batch,1), device=batch['aa'].device)
            t = t*(1-2 * self._interpolant_cfg.min_t) + self._interpolant_cfg.min_t
            if self.sample_structure:
                trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device) * self._interpolant_cfg.trans.sigma
                trans_0_c,_ = self.zero_center_part(trans_0,gen_mask,res_mask)
                trans_t = (1-t[...,None])*trans_0_c + t[...,None]*trans_1_c
                trans_t_c = torch.where(batch['generate_mask'][...,None],trans_t,trans_1_c)
                rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
                rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
                rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_t,rotmats_1)
                angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype)
                angles_t = torus.tor_geodesic_t(t[..., None], angles_1, angles_0)
                angles_t = torch.where(batch['generate_mask'][...,None],angles_t,angles_1)
                eps = torch.randn_like(x_local_1)
                x_local_t = t[...,None,None] * x_local_1 +  (1-t[...,None,None]) * eps
                x_local_t = torch.where(batch['generate_mask'][...,None,None],x_local_t,x_local_1)
                x_local_t = x_local_t * atom_mask[..., None]
                x_local_t[:,:,0:4]=x_local_1[:,:,0:4]
                if self.training:
                    bb_noise_scale = 0.05
                    bb_eps = torch.randn_like(x_local_1[:, :, 0:4]) * bb_noise_scale
                    bb_add_noise_mask = (torch.rand(num_batch, 1,1,1, device=x_local_1.device) < 0.5)
                    gen_mask_bb = batch['generate_mask'][..., None, None]
                    x_local_t[:, :, 0:4] = torch.where(
                        bb_add_noise_mask & gen_mask_bb,
                        x_local_1[:, :, 0:4] + bb_eps ,
                        x_local_1[:, :, 0:4]
                    )
            else:
                trans_t_c = trans_1_c.detach().clone()
                rotmats_t = rotmats_1.detach().clone()
                angles_t = angles_1.detach().clone()
            if self.sample_sequence:
                vq_0_simplex = self.k * torch.randn_like(vq_emb_simplex)
                vq_0_prob = F.softmax(vq_0_simplex,dim=-1)
                vq_t_simplex = ((1 - t[..., None]) * vq_0_simplex) + (t[..., None] * vq_emb_simplex)
                vq_t_simplex = torch.where(batch['generate_mask'][...,None],vq_t_simplex,vq_emb_simplex)
                vq_t_prob = F.softmax(vq_t_simplex,dim=-1)
                vq_t = sample_from(vq_t_prob)
                vq_t = torch.where(batch['generate_mask'],vq_t,vq_emb)


        pred_rotmats_1, pred_trans_1, pred_angles_1,pred_vq_1_prob   = self.joint_prediction_head(t, rotmats_t, trans_t_c, angles_t,vq_t, node_embed, edge_embed, gen_mask, res_mask)
        pred_vq_1 = sample_from(F.softmax(pred_vq_1_prob,dim=-1))
        pred_vq_1 = torch.where(batch['generate_mask'],pred_vq_1,torch.clamp(vq_emb,0,self.vq_size-1))

        pred_trans_1_c = pred_trans_1

        norm_scale = 1 / (1 - torch.min(t[...,None], torch.tensor(self._interpolant_cfg.t_normalization_clip)))
        x_global_t = local_to_global(
            x_local_t,
            trans_1.unsqueeze(2),
            rotmats_1.unsqueeze(2),
        )

        node_embe2 = self.node_embedder(
                batch['aa'], batch['res_nb'], batch['chain_nb'],
                x_global_t, atom_mask,
                structure_mask=batch['res_mask'], sequence_mask=batch['res_mask']
            )

        edge_embed2 = self.edge_embedder(
                batch['aa'], batch['res_nb'], batch['chain_nb'],
                x_global_t, atom_mask,
                structure_mask=batch['res_mask'], sequence_mask=batch['res_mask']
            )
        pred_x_global_1 = self.sidechain_decoder(
                                            x_global_t =x_global_t,
                                            timesteps = t,
                                            vq = vq_emb.reshape(num_batch,num_res),
                                            aa_index = batch['aa'],
                                            node_embed = node_embe2,
                                            edge_embed = edge_embed2,
                                            res_mask = batch['res_mask'],
                                            gen_mask = batch['generate_mask'],
                                            atom_mask = atom_mask)
        sidechain_loss,loss_dic = self.citation(pred_global_atoms = pred_x_global_1,
                                        true_atoms=trans_atoms_pos,
                                        atoms_mask=atom_mask,
                                        aa=batch['aa'],
                                        chi_angs =batch['torsion_angle'][:,:,1:],
                                        chi_masks = batch['torsion_angle_mask'][:,:,1:],
                                        res_mask=    batch['res_mask'],
                                        gen_mask=batch['generate_mask'])

        trans_loss = torch.sum((pred_trans_1_c - trans_1_c)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8)
        trans_loss = torch.mean(trans_loss)
        trans_loss_l2 = trans_l2_loss(pred_trans_1_c,gen_mask)
        trans_loss = trans_loss + trans_loss_l2

        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        rot_loss = torch.sum(((gt_rot_vf - pred_rot_vf) * norm_scale)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8)
        rot_loss = torch.mean(rot_loss)

        full_gt_atoms = all_atom.to_atom37(trans_1_c, rotmats_1)[:, :, :]
        full_pred_atoms = all_atom.to_atom37(pred_trans_1_c, pred_rotmats_1)[:, :, :]

        gt_bb_atoms = full_gt_atoms[:, :, :3]
        pred_bb_atoms = full_pred_atoms[:, :, :3]

        local_gt_bb_atoms= global_to_local(gt_bb_atoms,trans_1_c.unsqueeze(2),rotmats_1.unsqueeze(2))
        local_pred_bb_atoms= global_to_local(pred_bb_atoms,trans_1_c.unsqueeze(2), rotmats_1.unsqueeze(2))
        bb_atom_loss = torch.norm(
            local_gt_bb_atoms - local_pred_bb_atoms,dim=-1)
        bb_atom_loss = bb_atom_loss * gen_mask[..., None]
        bb_atom_loss = bb_atom_loss.mean(dim=-1).sum(-1) / (torch.sum(gen_mask,dim=-1) + 1e-8)
        bb_bond_loss = pep_bond_loss(full_pred_atoms,full_gt_atoms,gen_mask).mean()
        bb_atom_loss = torch.mean(bb_atom_loss)+2*bb_bond_loss
        pred_seqs_1_prob = torch.matmul(F.softmax(pred_vq_1_prob,dim=-1),self.vq_to_seq_matrix.to(pred_vq_1_prob.device))
        seqs_loss = F.nll_loss(
                    torch.log(pred_seqs_1_prob.clamp_min(1e-8)).view(-1, pred_seqs_1_prob.shape[-1]),
                    torch.clamp(seqs_1, 0, 19).view(-1),
                    reduction="none"
                ).view(pred_seqs_1_prob.shape[:-1])
        seqs_loss = torch.sum(seqs_loss * gen_mask, dim=-1) / (torch.sum(gen_mask,dim=-1) + 1e-8)
        seqs_loss = torch.mean(seqs_loss)
        vq_loss = time_dependent_vq_loss(
                            vq_logits=pred_vq_1_prob,
                            target_vq=vq_emb,
                            gen_mask=gen_mask,
                            t=t,
                            vq_size=self.vq_size,
                            eps_min=0.02,
                            eps_max=0.30,
                            mode="same_aa",
                            aa_target=seqs_1,
                            vq_to_seq_matrix=self.vq_to_seq_matrix,
                            reduce=False,
                        )

        vq_loss = torch.sum(vq_loss * gen_mask, dim=-1) / (torch.sum(gen_mask,dim=-1) + 1e-8)
        vq_loss = torch.mean(vq_loss)

        gt_angle_vf = torus.tor_logmap(angles_t[:,:,:1], angles_1[:,:,:1])
        gt_angle_vf_vec = torch.cat([torch.sin(gt_angle_vf),torch.cos(gt_angle_vf)],dim=-1)
        pred_angle_vf = torus.tor_logmap(angles_t[:,:,:1], pred_angles_1[:,:,:1])
        pred_angle_vf_vec = torch.cat([torch.sin(pred_angle_vf),torch.cos(pred_angle_vf)],dim=-1)

        angle_loss = torch.sum((((gt_angle_vf_vec - pred_angle_vf_vec) * norm_scale)**2*gen_mask.unsqueeze(-1)),dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1)*2 + 1e-8)

        angle_loss = torch.mean(angle_loss)


        angles_1_vec = torch.cat([torch.sin(angles_1[:,:,:1]),torch.cos(angles_1[:,:,:1])],dim=-1)
        pred_angles_1_vec = torch.cat([torch.sin(pred_angles_1[:,:,:1]),torch.cos(pred_angles_1[:,:,:1])],dim=-1)
        torsion_loss = torch.sum(((pred_angles_1_vec - angles_1_vec)**2*gen_mask.unsqueeze(-1)),dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1)*2 + 1e-8)
        torsion_loss = torch.mean(torsion_loss)
        lossdic = {k:v.item() for k,v in loss_dic.items()}
        lossdic['trans_loss'] = trans_loss.item()
        lossdic['trans_l2_loss'] = trans_loss_l2.item()
        lossdic['rot_loss'] = rot_loss.item()
        lossdic['bb_atom_loss'] = bb_atom_loss.item()
        lossdic['seqs_loss'] = seqs_loss.item()
        lossdic['angle_loss'] = angle_loss.item()
        lossdic['torsion_loss'] = torsion_loss.item()
        lossdic['vq_loss'] = vq_loss.item()
        lossdic['bb_bond_loss'] = bb_bond_loss.item()
        lossdic['sidechain_loss'] = sidechain_loss.item()
        return {
            'sidechain_loss': sidechain_loss,
            "trans_loss": trans_loss,
            'rot_loss': rot_loss,
            'bb_atom_loss': bb_atom_loss,
            'seqs_loss': seqs_loss,
            'angle_loss': angle_loss,
            'torsion_loss': torsion_loss,
            'vq_loss': vq_loss,
        },lossdic

    @torch.no_grad()
    def sample_step2(self,batch,pred_trans_1,pred_rotmats_1,pred_seq,vq_emb,num_steps = 100,centerlize=False,updata_mode= 'alpha',gen_mask=None):
        num_batch, num_res = batch['aa'].shape
        res_mask = batch['res_mask']
        if gen_mask is None:
            gen_mask = batch['generate_mask']
        _,center = self.zero_center_part(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],gen_mask,res_mask)
        ts = torch.linspace(1.e-2, 1.0, num_steps)
        t_1 = ts[0]
        clean_traj = []
        pos_bb = reconstruct_backbone(R=pred_rotmats_1,t=pred_trans_1,aa=pred_seq,chain_nb=batch['chain_nb'],res_nb=batch['res_nb'],mask=batch['res_mask'])
        pos_ha = F.pad(pos_bb, pad=(0,0,0,15-4), value=0.)
        backbone_atoms = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
        if centerlize:
            backbone_atoms = self.center_initialize(backbone_atoms ,center,res_mask,rot=False)
            pred_trans_1 = self.center_initialize(pred_trans_1.reshape(num_batch,num_res,1,3),center,res_mask,rot=False).reshape(num_batch,num_res,3)
        local_backbone_atoms = global_to_local(backbone_atoms,pred_trans_1.unsqueeze(2),pred_rotmats_1.unsqueeze(2))

        mask_new = get_heavyatom_mask(pred_seq)
        aa_new = pred_seq
        eps = torch.randn((num_batch,num_res,15,3), device=batch['aa'].device)

        x_local_0= local_backbone_atoms
        x_local_0[:,:,4:] = torch.where(batch['generate_mask'][:,:,None,None],eps[:,:,4:],x_local_0[:,:,4:])
        x_local_1 = x_local_0

        x_global_1 = local_to_global(
                x_local_1,
                pred_trans_1.unsqueeze(2),
                pred_rotmats_1.unsqueeze(2),
            )
        for t_2 in ts[1:]:
            t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
            node_embe2 = self.node_embedder(
                pred_seq, batch['res_nb'], batch['chain_nb'],
                x_global_1, mask_new ,
                structure_mask=res_mask, sequence_mask=res_mask
            )

            edge_embed2 = self.edge_embedder(
                pred_seq, batch['res_nb'], batch['chain_nb'],
                x_global_1, mask_new,
                structure_mask=res_mask, sequence_mask=res_mask
            )
            pred_x_global_1 = self.sidechain_decoder(
                                            x_global_t =x_global_1,
                                            timesteps = t,
                                            vq = vq_emb.reshape(num_batch,num_res),
                                            aa_index = pred_seq,
                                            node_embed = node_embe2,
                                            edge_embed = edge_embed2,
                                            res_mask = batch['res_mask'],
                                            gen_mask = batch['generate_mask'],
                                            atom_mask = mask_new)
            alpha = (t_2 - t_1) / (1.0 - t_1 + 1e-8).to(batch['aa'].device)
            d_t = (t_2-t_1) * torch.ones((num_batch, 1), device=batch['aa'].device)
            if updata_mode == 'alpha':
                updata = alpha
            elif updata_mode == 'd_t':
                updata = d_t
            pred_x_global_1 =pred_x_global_1.reshape(num_batch,num_res,15,3)
            x_global_2 = x_global_1+ (pred_x_global_1-x_global_1)*updata[...,None,None]
            x_global_2 = torch.where(batch['generate_mask'][...,None,None],x_global_2,backbone_atoms)
            x_global_2[:,:,0:4] = backbone_atoms[:,:,0:4]
            x_global_1 = x_global_2
            clean_traj.append(x_global_2.detach().clone().cpu())
            t_1 = t_2
        t_1 = ts[-1]
        t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1

        node_embe2 = self.node_embedder(
            pred_seq, batch['res_nb'], batch['chain_nb'],
            x_global_1, mask_new ,
            structure_mask=res_mask, sequence_mask=res_mask
        )

        edge_embed2 = self.edge_embedder(
            pred_seq, batch['res_nb'], batch['chain_nb'],
            x_global_1, mask_new,
            structure_mask=res_mask, sequence_mask=res_mask
        )
        pred_x_global_1 = self.sidechain_decoder(
                                            x_global_t =x_global_1,
                                            timesteps = t,
                                            vq = vq_emb.reshape(num_batch,num_res),
                                            aa_index = pred_seq,
                                            node_embed = node_embe2,
                                            edge_embed = edge_embed2,
                                            res_mask = batch['res_mask'],
                                            gen_mask = batch['generate_mask'],
                                            atom_mask = mask_new)
        pred_x_global_1= torch.where(batch['generate_mask'][...,None,None],pred_x_global_1,backbone_atoms)
        pred_x_global_1[:,:,0:4] = backbone_atoms[:,:,0:4]
        clean_traj.append({'pos':pred_x_global_1.detach().cpu(),'trans':pred_trans_1.detach().cpu(),'rot':pred_rotmats_1.detach().cpu(),'seq':pred_seq.detach().cpu(),'seq_mask':mask_new.detach().cpu()})
        return clean_traj
    @torch.no_grad()
    def sample(self, batch, num_steps = 100, sample_bb=True, sample_ang=True, sample_seq=True,updata_mode= 'alpha'):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask = batch['generate_mask'],batch['res_mask']
        K = self._interpolant_cfg.seqs.num_classes
        k = self._interpolant_cfg.seqs.simplex_value
        angle_mask_loss = torsions_mask.to(batch['aa'].device)
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed,vq_emb = self.encode(batch)
        trans_1_c,center = self.zero_center_part(trans_1,gen_mask,res_mask)
        vq_emb = vq_emb.reshape(num_batch,-1)
        vq_emb_simplex = (clampped_one_hot(vq_emb, self.vq_size).float() * self.k * 2 - self.k)

        if sample_bb:
            rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
            rotmats_0 = torch.where(batch['generate_mask'][...,None,None],rotmats_0,rotmats_1)
            trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device)
            trans_0_c,center = self.zero_center_part(trans_0,gen_mask,res_mask)
            trans_0_c = torch.where(batch['generate_mask'][...,None],trans_0_c,trans_1_c)
        else:
            rotmats_0 = rotmats_1.detach().clone()
            trans_0_c = trans_1_c.detach().clone()
        if sample_ang:
            angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype)
            angles_0 = torch.where(batch['generate_mask'][...,None],angles_0,angles_1)
        else:
            angles_0 = angles_1.detach().clone()
            vq_t_2 = vq_emb.detach().clone()
            vq_t_2_simplex = vq_emb_simplex.detach().clone()
        if sample_seq:
            vq_0_simplex = k * torch.randn((num_batch,num_res,self.vq_size), device=batch['aa'].device)
            vq_0_prob = F.softmax(vq_0_simplex,dim=-1)
            vq_0 = sample_from(vq_0_prob)
            vq_0 = torch.where(batch['generate_mask'],vq_0,vq_emb)
            vq_0_simplex = torch.where(batch['generate_mask'][...,None],vq_0_simplex,vq_emb_simplex)
        else:
            vq_matrix =  torch.matmul(F.one_hot(seqs_1,22).float(), self.seq_to_vq_matrix.to(batch['aa'].device))
            vq_0_simplex =vq_matrix* self.k * 2 - self.k
            vq_0_prob = F.softmax(vq_0_simplex,dim=-1)
            vq_0 = sample_from(vq_0_prob)
            vq_0 = torch.where(batch['generate_mask'],vq_0,vq_emb)
            vq_0_simplex = torch.where(batch['generate_mask'][...,None],vq_0_simplex,vq_emb_simplex)
            seqs_1_simplex = self.seq_to_simplex(seqs_1)


        ts = torch.linspace(1.e-2, 1.0, num_steps)
        t_1 = ts[0]
        clean_traj = []
        rotmats_t_1, trans_t_1_c, angles_t_1,vq_t_1,vq_t_1_simplex = rotmats_0, trans_0_c, angles_0,vq_0,vq_0_simplex


        for t_2 in ts[1:]:

            t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
            pred_rotmats_1, pred_trans_1, pred_angles_1,pred_vq_1_prob    = self.joint_prediction_head(t, rotmats_t_1, trans_t_1_c, angles_t_1,vq_t_1, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
            pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
            pred_trans_1_c = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_1_c)
            pred_angles_1 = torch.where(batch['generate_mask'][...,None],pred_angles_1,angles_1)
            if sample_seq:
                pred_vq_1 = sample_from(F.softmax(pred_vq_1_prob,dim=-1))
            else:
                pred_vq_1 = sample_from(F.softmax(pred_vq_1_prob*vq_matrix,dim=-1))
            pred_vq_1_prob = F.softmax(pred_vq_1_prob, dim=-1)
            pred_vq_1_simplex_soft = pred_vq_1_prob * (2 * self.k) - self.k

            d_t = (t_2-t_1) * torch.ones((num_batch, 1), device=batch['aa'].device)
            alpha = (t_2 - t_1) / (1.0 - t_1 + 1e-8)
            alpha = alpha.to(batch['aa'].device)
            if updata_mode == 'alpha':
                updata = alpha
            elif updata_mode == 'd_t':
                updata = d_t
            trans_t_2 = trans_t_1_c + updata[..., None] * (pred_trans_1_c - trans_t_1_c)
            trans_t_2_c = torch.where(batch['generate_mask'][...,None],trans_t_2,trans_1_c)
            rotmats_t_2 = so3_utils.geodesic_t(updata[..., None], pred_rotmats_1, rotmats_t_1)
            rotmats_t_2 = torch.where(batch['generate_mask'][...,None,None],rotmats_t_2,rotmats_1)
            angles_t_2 = torus.tor_geodesic_t(updata[..., None],pred_angles_1, angles_t_1)
            angles_t_2 = torch.where(batch['generate_mask'][...,None],angles_t_2,angles_1)


            vq_t_2_simplex = vq_t_1_simplex + updata[..., None] * (pred_vq_1_simplex_soft - vq_t_1_simplex)
            vq_t_2 = sample_from(F.softmax(vq_t_2_simplex,dim=-1))
            vq_t_2 = torch.where(batch['generate_mask'],vq_t_2,vq_emb)
            pred_seqs_1 = torch.matmul(F.one_hot(vq_t_2,self.vq_size).float(),self.vq_to_seq_matrix.to(batch['aa'].device)).argmax(dim=-1)
            pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
            pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
            torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1)
            pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
            if not sample_bb:
                pred_trans_1_c = trans_1_c.detach().clone()
                pred_rotmats_1 = rotmats_1.detach().clone()
            if not sample_ang:
                pred_angles_1 = angles_1.detach().clone()
                pred_vq_1 = vq_emb.detach().clone()
                pred_vq_1_simplex_soft = vq_emb_simplex.detach().clone()
            if not sample_seq:
                pred_seqs_1 = seqs_1.detach().clone()
                pred_seqs_1_simplex = seqs_1_simplex.detach().clone()
            clean_traj.append({'rotmats':pred_rotmats_1.cpu(),
                                'trans':pred_trans_1_c.cpu(),
                                'angles':pred_angles_1.cpu(),
                                'seqs':pred_seqs_1.cpu(),
                                'seqs_simplex':pred_seqs_1_simplex.cpu(),
                                'vq':pred_vq_1.cpu(),
                                'vq_simplex':pred_vq_1_simplex_soft.cpu(),
                                    'rotmats_1':rotmats_1.cpu(),'trans_1':trans_1_c.cpu(),'angles_1':angles_1.cpu(),'seqs_1':seqs_1.cpu()})

            if not sample_bb:
                trans_t_2_c = trans_1_c.detach().clone()
                rotmats_t_2 = rotmats_1.detach().clone()
            if not sample_ang:
                angles_t_2 = angles_1.detach().clone()
                vq_t_2 = vq_emb.detach().clone()
            if not sample_seq:
                seqs_t_2 = seqs_1.detach().clone()
            rotmats_t_1, trans_t_1_c, angles_t_1, vq_t_1, vq_t_1_simplex = rotmats_t_2, trans_t_2_c, angles_t_2, vq_t_2, vq_t_2_simplex
            t_1 = t_2

        t_1 = ts[-1]
        t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
        pred_rotmats_1, pred_trans_1, pred_angles_1,pred_vq_1_prob    = self.joint_prediction_head(t, rotmats_t_1, trans_t_1_c, angles_t_1,vq_t_1, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
        pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
        pred_trans_1_c = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_1_c)
        pred_angles_1 = torch.where(batch['generate_mask'][...,None],pred_angles_1,angles_1)
        if sample_seq:
                pred_vq_1 = sample_from(F.softmax(pred_vq_1_prob,dim=-1))
        else:
                pred_vq_1 = sample_from(F.softmax(pred_vq_1_prob*vq_matrix,dim=-1))
        pred_vq_1 = torch.where(batch['generate_mask'],pred_vq_1,vq_emb)
        pred_vq_1_simplex = clampped_one_hot(pred_vq_1.squeeze(), self.vq_size).float() * self.k * 2 - self.k
        pred_seqs_1 = torch.matmul(F.one_hot(pred_vq_1,self.vq_size).float(),self.vq_to_seq_matrix.to(batch['aa'].device)).argmax(dim=-1)
        pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
        pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
        torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1)
        pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
        if not sample_bb:
            pred_trans_1_c = trans_1_c.detach().clone()
            pred_rotmats_1 = rotmats_1.detach().clone()
        if not sample_ang:
            pred_angles_1 = angles_1.detach().clone()
            pred_vq_1 = vq_emb.detach().clone()
            pred_vq_1_simplex = vq_emb_simplex.detach().clone()
        if not sample_seq:
            pred_seqs_1 = seqs_1.detach().clone()
            pred_seqs_1_simplex = seqs_1_simplex.detach().clone()
        clean_traj.append({'rotmats':pred_rotmats_1.cpu(),
        'trans':pred_trans_1_c.cpu(),'angles':pred_angles_1.cpu(),'seqs':pred_seqs_1.cpu(),
         'seqs_simplex':pred_seqs_1_simplex.cpu(),'vq':pred_vq_1.cpu(),
         'vq_emb_simplex':pred_vq_1_simplex.cpu(),
                                'rotmats_1':rotmats_1.cpu(),
                                'trans_1':trans_1_c.cpu(),
                                'angles_1':angles_1.cpu(),
                                'seqs_1':seqs_1.cpu(),
                                'vq_1':vq_emb.cpu(),
                                'vq_1_simplex':vq_emb_simplex.cpu()})

        return clean_traj
