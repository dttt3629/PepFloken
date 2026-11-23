import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import math
from tqdm.auto import tqdm
import functools
import os
import argparse
import pandas as pd
from copy import deepcopy

from models_con.pep_dataloader import PepDataset

from pepfloken.utils.misc import load_config
from pepfloken.utils.train import recursive_to

from pepfloken.modules.common.geometry import reconstruct_backbone, reconstruct_backbone_partially, align, batch_align
from pepfloken.modules.protein.writers import save_pdb

from pepfloken.utils.data import PaddingCollate

from models_con.utils import process_dic

import gc
from Bio.PDB.StructureBuilder import StructureBuilder
from con_code.common.residue_con import atom_element
from Bio.PDB import PDBIO
from models_con.pepfloken import PepFloken
from Bio.PDB import PDBParser
from collections import defaultdict
from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask
def built_pdb(dicts,icode=None,save_path=None):
    builder = StructureBuilder()
    structure = builder.init_structure(0)  
    builder.init_model(0)
    for ch,resls in dicts.items():
        builder.init_chain(ch)
        builder.init_seg('    ')
        resn = 0
        for res in resls:
            for res_name, atoms in res.items():
                resn += 1
                builder.init_residue(
                resname = res_name,
                field = ' ',
                resseq = resn,
                icode = icode[resn-1] if icode is not None else ' ',
            )
                for atom_name, coords in atoms.items():
                    if len(atom_name) == 1: fullname = ' %s  ' % atom_name
                    elif len(atom_name) == 2: fullname = ' %s ' % atom_name
                    elif len(atom_name) == 3: fullname = ' %s' % atom_name
                    builder.init_atom(atom_name, coords.tolist(), 0.0, 1.0, ' ', fullname=fullname,element=atom_element[atom_name])
    structure = builder.get_structure()
    if save_path is not None:
        io = PDBIO()
        io.set_structure(structure)
        io.save(save_path)
    return structure
def gen_res(resseq, chain_nb):
    counts = defaultdict(int)
    codes = []
    for chain, number in zip(chain_nb.tolist(), resseq.tolist()):
        key = (chain, number)
        occurrence = counts[key]
        if occurrence > 26:
            raise ValueError(f'Too many residues with number {number} in chain {chain}')
        codes.append(' ' if occurrence == 0 else chr(ord('A') + occurrence - 1))
        counts[key] += 1
    return codes

def built_pdb2(pos,resls,ch,save_path=None):
    builder = StructureBuilder()
    structure = builder.init_structure(0)  
    builder.init_model(0)
    builder.init_chain(ch)
    builder.init_seg('    ')
    resn = 0
    for n,res_name in enumerate(resls):
        resn += 1
        builder.init_residue(
        resname = res_name,
        field = ' ',
        resseq = resn,
        icode = ' ',
        )
        atoms = restype_name_to_atom14_names[res_name]
        for i in range(len(atoms)):
            if atoms[i] == '': 
                break
            atom_name = atoms[i]
            coords = pos[n][i]
            if len(atom_name) == 1: fullname = ' %s  ' % atom_name
            elif len(atom_name) == 2: fullname = ' %s ' % atom_name
            elif len(atom_name) == 3: fullname = ' %s' % atom_name
            builder.init_atom(atom_name, coords.tolist(), 0.0, 1.0, ' ', fullname=fullname,element=atom_element[atom_name])
    structure = builder.get_structure()
    if save_path is not None:
        io = PDBIO()
        io.set_structure(structure)
        io.save(save_path)
    return structure
collate_fn = PaddingCollate(eight=False)
def save_samples_bb(samples,save_dir,init_num=0):
    batch = recursive_to(samples['batch'],'cpu')
    chain_id = [list(item) for item in zip(*batch['chain_id'])][0] 
    icode = gen_res(batch['resseq'][0], batch['chain_nb'][0])
    nums = len(batch['id'])
    id = batch['id'][0]
    pos_bb = reconstruct_backbone(R=samples['rotmats'],t=samples['trans'],aa=samples['seqs'],chain_nb=batch['chain_nb'],res_nb=batch['res_nb'],mask=batch['res_mask']) 
    pos_ha = F.pad(pos_bb, pad=(0,0,0,15-4), value=0.) 
    pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
    mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
    mask_bb_atoms[:,:,:4] = True
    mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])
    aa_new = samples['seqs']
    for i in range(nums):
        data_saved = {
                      'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                      'aa':aa_new[i], 'mask_heavyatom':mask_new[i], 'pos_heavyatom':pos_new[i],
                    }
        save_pdb(data_saved,path=os.path.join(save_dir,f'sample_{i+init_num}.pdb'))
    data_saved = {
                    'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                    'aa':batch['aa'][0], 'mask_heavyatom':batch['mask_heavyatom'][0], 'pos_heavyatom':batch['pos_heavyatom'][0],
                }
    save_pdb(data_saved,path=os.path.join(save_dir,f'gt.pdb'))

def store_from_trajls(traj_list,ind,k):
    out = [ ]
    for traj in traj_list:
        if ind ==None:
            out.append(traj[k])
        else:
            out.append(traj[ind][k])
    return out
import argparse
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from con_code.common.residue_con import atom_types,resatoms,restype_name_to_atom14_names, chi_angles_atoms,three_to_one,three2idx,three2self
from con_code.utils.structure_cal import obtain_dihediral_angles,_get_torsion
from con_code.script import get_model,get_pdb_from_backbone,get_decode_from_aa_psi
from models_con.convq import Convq
from con_code.model.vqvae import Con_CustomVQVAE
from pepfloken.modules.common.geometry import construct_3d_basis
from pepfloken.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms
def zero_center_part(pos,gen_mask,res_mask):

        center = torch.sum(pos * gen_mask[...,None], dim=1) / (torch.sum(gen_mask,dim=-1,keepdim=True) + 1e-8) 
        center = center.unsqueeze(1) 
        pos = pos - center
        pos = pos * res_mask[...,None]
        return pos,center
def export_pdb(readout,batch,save_dir,childdir = None,align = False, relax=None):
    save_dir2 = os.path.join(save_dir,batch['id'][0])+'.pt'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
    if childdir is not None:
        save_dir2 = os.path.join(save_dir2,childdir)
        if not os.path.exists(save_dir2):
            os.makedirs(save_dir2)
    num = 0
    if align:
        _,center = zero_center_part(batch['pos_heavyatom'][:,:,BBHeavyAtom.CA,:].cpu(),batch['generate_mask'].cpu(),batch['res_mask'].cpu())
    for taj in readout:

        pos =  taj[-1]['pos'].cpu()
        if align:
            pos = pos + center.unsqueeze(1)
        trans = taj[-1]['trans'].cpu()
        rot = taj[-1]['rot'].cpu()
        seq = taj[-1]['seq'].cpu()
        seq_mask = taj[-1]['seq_mask'].cpu()
        chain_id = [list(item) for item in zip(*batch['chain_id'])][0] 
        icode = gen_res(batch['resseq'][0], batch['chain_nb'][0])
        nums = pos.shape[0]

        for i in range(nums):

            data_saved = {
                        'chain_nb':batch['chain_nb'][0].cpu(),'chain_id':chain_id,'resseq':batch['resseq'][0].cpu(),'icode':icode,
                        'aa':seq[i], 'mask_heavyatom':seq_mask[i], 'pos_heavyatom':pos[i],
                        }
            sample_path = os.path.join(save_dir2, f'sample_{num}.pdb')
            save_pdb(data_saved, path=sample_path)
            if relax is not None:
                relax(sample_path)
            num += 1
    data_saved = {
                    'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                    'aa':batch['aa'][0], 'mask_heavyatom':batch['mask_heavyatom'][0], 'pos_heavyatom':batch['pos_heavyatom'][0],
                }
    save_pdb(data_saved,path=os.path.join(save_dir2,f'gt.pdb'))
def export_pdb2(traj,batch,save_dir,childdir = None,align = False):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num = 0


    pos =  traj[-1]['pos'].cpu()
    trans = traj[-1]['trans'].cpu()
    rot = traj[-1]['rot'].cpu()
    seq = traj[-1]['seq'].cpu()
    seq_mask = traj[-1]['seq_mask'].cpu()
    res_mask = batch['res_mask'].cpu()
    if align:
        _,center = zero_center_part(batch['pos_heavyatom'][:,:,BBHeavyAtom.CA,:].cpu(),batch['generate_mask'].cpu(),batch['res_mask'].cpu())
        pos = pos + center.unsqueeze(1)
    for i in range(pos.shape[0]):
        save_dir2 = os.path.join(save_dir,batch['id'][i])+'.pt'
        if childdir is not None:
            save_dir2 = os.path.join(save_dir2,childdir)
            if not os.path.exists(save_dir2):
                os.makedirs(save_dir2)
        chain_id = [list(item) for item in zip(*batch['chain_id'])][i] 
        chain_id =  [x for x in chain_id if x not in ' ']
        icode = gen_res(batch['resseq'][i][res_mask[i]], batch['chain_nb'][i][res_mask[i]])
        nums = pos.shape[0]

        data_saved = {
                    'chain_nb':batch['chain_nb'][i].cpu()[res_mask[i]], 'chain_id':chain_id,'resseq':batch['resseq'][i].cpu()[res_mask[i]],'icode':icode,
                    'aa':seq[i][res_mask[i]], 'mask_heavyatom':seq_mask[i][res_mask[i]], 'pos_heavyatom':pos[i][res_mask[i]],
                    }
        save_pdb(data_saved,path=os.path.join(save_dir2,f'sample.pdb'))

        data_saved = {
                        'chain_nb':batch['chain_nb'][i][res_mask[i]], 'chain_id':chain_id,'resseq':batch['resseq'][i][res_mask[i]],'icode':icode,
                        'aa':batch['aa'][i][res_mask[i]], 'mask_heavyatom':batch['mask_heavyatom'][i][res_mask[i]], 'pos_heavyatom':batch['pos_heavyatom'][i][res_mask[i]],
                    }
        save_pdb(data_saved,path=os.path.join(save_dir2,f'gt.pdb'))
def calc_full_atom_rmsd_from_traj(traj, batch, centerlize=False, gen_only=True, eps=1e-8):

    pred_pos = traj[-1]['pos'].to(device)
    true_pos = batch['pos_heavyatom']

    pred_mask = traj[-1].get('seq_mask', batch['mask_heavyatom']).bool().to(device)
    true_mask = batch['mask_heavyatom'].bool()

    if centerlize:
        _, center = zero_center_part(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA, :],
            batch['generate_mask'],
            batch['res_mask'],
        )
        pred_pos = pred_pos + center.unsqueeze(1)

    atom_mask = pred_mask & true_mask

    if gen_only:
        atom_mask = atom_mask & batch['generate_mask'].bool()[..., None]
    else:
        atom_mask = atom_mask & batch['res_mask'].bool()[..., None]

    diff2 = ((pred_pos - true_pos) ** 2).sum(dim=-1)  

    sq_sum = (diff2 * atom_mask.float()).sum()
    atom_num = atom_mask.float().sum()

    rmsd = torch.sqrt(sq_sum / (atom_num + eps))

    return rmsd, sq_sum.detach(), atom_num.detach()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='./configs/train.yaml')
    args.add_argument('--device', type=str, default='cuda:0')
    args.add_argument('--ckpt', type=str, help='Override the checkpoint specified in the configuration.')
    args.add_argument('--output', type=str, default='./result/pepfloken/')
    args.add_argument('--num_steps', type=int, default=200)
    args.add_argument('--num_samples', type=int, default=40)
    args.add_argument('--batch_size', type=int, default=40)
    args.add_argument("--n_hiddens", type=int, default=256)
    args.add_argument("--n_embeddings", type=int, default=1024)
    args.add_argument("--centerlize", type=bool, default=False)
    args.add_argument('--skip-relax', action='store_true', help='Write sampled PDBs without PyRosetta FastRelax.')
    parser = args.parse_args()
    relax = None
    if not parser.skip_relax:
        try:
            from con_code.utils.eval import RosettaInterfaceScorer
        except ModuleNotFoundError as exc:
            if exc.name != 'pyrosetta':
                raise
            args.error('PyRosetta is required for FastRelax. Install it or pass --skip-relax.')
        relax = RosettaInterfaceScorer().relax_in_place
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
    config, config_name = load_config(parser.config)
    config['device'] = parser.device
    config.model['vq_tokendic'] = vq_tokendic
    config.model['vq_to_seq_matrix'] = vq_to_seq_matrix
    config.model['seq_to_vq_matrix'] = seq_to_vq_matrix
    device = parser.device
    dataset = PepDataset(structure_dir = config.dataset.test.structure_dir, dataset_dir = config.dataset.test.dataset_dir,
                                            name = config.dataset.test.name, transform=None, reset=config.dataset.test.reset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=4, pin_memory=True)
    checkpoint_path = parser.ckpt or config.get('sampling', {}).get('checkpoint', './ckpt/pepfloken.pt')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    dataloader2 = DataLoader(dataset, batch_size=parser.batch_size, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=4, pin_memory=True)

    model = PepFloken(config.model).to(device)
    model.load_state_dict(process_dic(ckpt['model']))
    model.eval()
    dic = {'id':[],'len':[],'tran':[],'aar':[],'rot':[],'trans_loss':[],'rot_loss':[],'sidechain_loss':[]}
    num_it = parser.num_samples // parser.batch_size

    with torch.no_grad():
        torch.cuda.empty_cache()
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            traj_list = []
            readout1 = []
            readout2 = []
            for j in range(num_it):
                data_list = [deepcopy(item) for _ in range(parser.batch_size)]
                batch = recursive_to(collate_fn(data_list),device)
                loss_dic,loss_dic2 = model(batch)

                traj_1 = model.sample(batch,num_steps=parser.num_steps)
                traj_list.append(traj_1)
                rotmats_2 = traj_1[-1]['rotmats'].to(device)
                trans_2 = traj_1[-1]['trans'].to(device)
                aals = traj_1[-1]['seqs'].to(device)
                vq_emb = traj_1[-1]['vq'].to(device)
                traj_3 = model.sample_step2(batch,trans_2,rotmats_2,aals,vq_emb,num_steps=parser.num_steps,centerlize=parser.centerlize,updata_mode='alpha')
                readout2.append(traj_3)

            export_pdb(readout2,batch,save_dir=f'{parser.output}sample/',childdir='full_eval',align = parser.centerlize,relax=relax)
            ca_dist = torch.sqrt(torch.sum((torch.concat(store_from_trajls(traj_list,-1,'trans'))-torch.concat(store_from_trajls(traj_list,-1,'trans_1')))**2*torch.concat([batch['generate_mask'][...,None].cpu().long() for _ in range(num_it)],dim=0)) / (torch.sum(batch['generate_mask'])*2 + 1e-8).cpu()) 
            rot_dist = torch.sqrt(torch.sum((torch.concat(store_from_trajls(traj_list,-1,'rotmats'))-torch.concat(store_from_trajls(traj_list,-1,'rotmats_1')))**2*torch.concat([batch['generate_mask'][...,None,None].long().cpu() for _ in range(num_it)],dim=0)) / (torch.sum(batch['generate_mask'])*2 + 1e-8).cpu()) 
            aar = torch.concat(store_from_trajls(traj_list,-1,'seqs'))==torch.concat(store_from_trajls(traj_list,-1,'seqs_1'))
            aar = torch.sum(aar* batch['generate_mask'][:1].long().cpu())/ (torch.sum(batch['generate_mask']) + 1e-8).cpu()/num_it
            vqr =torch.concat(store_from_trajls(traj_list,-1,'vq'))==torch.concat(store_from_trajls(traj_list,-1,'vq_1'))* torch.concat([batch['generate_mask'].long().cpu() for _ in range(num_it)],dim=0)
            vqr = torch.sum(vqr* batch['generate_mask'][:1].long().cpu()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu()/num_it


            print(f'tran:{ca_dist},rot:{rot_dist},aar:{aar},vqr:{vqr},len:{batch["generate_mask"].sum().item()}')


            torch.cuda.empty_cache()
            gc.collect()

            dic['tran'].append(ca_dist.item())
            dic['rot'].append(rot_dist.item())
            dic['aar'].append(aar.item())
            dic['trans_loss'].append(loss_dic['trans_loss'].item())
            dic['rot_loss'].append(loss_dic['rot_loss'].item())
            dic['sidechain_loss'].append(loss_dic['sidechain_loss'].item())
            dic['id'].append(batch['id'][0])
            dic['len'].append(batch['generate_mask'].sum().item())
            for traj in traj_list:
                traj[-1]['batch'] = batch
            if not os.path.exists(f'{parser.output}output/'):
                os.makedirs(f'{parser.output}output/')
            torch.save([traj[-1] for traj in traj_list],f'{parser.output}output/{batch["id"][0]}.pt')
            del batch
    dic = pd.DataFrame(dic)

