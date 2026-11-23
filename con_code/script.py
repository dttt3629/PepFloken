import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
import numpy as np
from con_code.common.residue_con import atom_types,resatoms,restype_name_to_atom14_names, chi_angles_atoms,three_to_one,three2idx,three2self,chi_angles_ind_37

import torch
import warnings
from con_code.protokenizer import proteintokenizer
from con_code.model.vqvae import Con_CustomVQVAE
from con_code.dataset.vae_dataset import vae_dataset,pep_collate_fn
import numpy as np
from torch.nn import functional as F
from torch import nn
import argparse
from con_code.common.residue_con import atom_types,resatoms,restype_name_to_atom14_names, chi_angles_atoms,three_to_one,three2idx,three2self,chi_angles_atoms,amino_acid_params,rigid_group_atom_positions,rigid_atoms37_position,chi_angles_mask_dic
from con_code.utils.structure_cal import obtain_dihediral_angles,_get_torsion,cos_sin_to_degrees,rotate_around_axis
from con_code.utils.structure_cal import rotate_point_around_axis,solve_target_vector_with_norm,align_points,align_points_batch
from con_code.model.vqvae import pads
from Bio.PDB.StructureBuilder import StructureBuilder
from con_code.common.residue_con import atom_element
from Bio.PDB import PDBIO

def built_pdb(dicts,save_path=None):
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
                icode = ' ',
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
def paraser_protein(path,chain = None):
     u = mda.Universe(path,topology_format='PDB') 
     if chain is not None:
          protein = u.select_atoms(f'protein and chainid {chain}')
     else:
          protein = u.select_atoms('protein')
     p_ca,p_n,p_c = [],[],[]
     resls = list(protein.residues)
     phi_list=[]
     pure_res_lis, seq, node_s, X_ca, X_n, X_c = [], [], [], [], [], []
     resid = []
     segid = []
     angls =[]
     chi_angs = []
     chi_masks = []
     atoms37 =[]
     only_mainchain =[]
     for i, res in enumerate(resls):

          res_id = res.resid  
          res_name = res.resname.strip()
          if res_name not in three_to_one.keys():
                    continue
          try:
               N = res.atoms.select_atoms('name N')[0].position
               CA = res.atoms.select_atoms('name CA')[0].position
               C = res.atoms.select_atoms('name C')[0].position
               O = res.atoms.select_atoms("name O")[0].position
          except:
               continue

          pos37 = np.zeros((37,3))
          for a in res.atoms:
               if a.name in atom_types:
                    pos37[atom_types.index(a.name)]=a.position
          ang = obtain_dihediral_angles(res) 
          seg = res.segid
          seq.append(three2idx[three2self.get(res_name, 'X')])
          X_ca.append(CA)
          X_n.append(N)
          X_c.append(C)
          angls.append(ang)
          resid.append((res.segid,res_name,res.resid))
          atoms37.append(pos37)
          chi_ang= [0,0,0,0]
          chi_mask = [0,0,0,0]
          flag= False
          atomsls = [a.name for a in res.atoms]
          for ai in resatoms[res_name]:
               if ai not in atomsls:
                    flag = True
          if flag:
               only_mainchain.append(1)
          else:
               only_mainchain.append(0)
               agnles = chi_angles_atoms[res_name]

               for agi,ag in enumerate(agnles):
                    chi_mask[agi] = 1
                    pos = [torch.tensor(pos37[atom_types.index(z)]) for z in ag]
                    for pos1 in pos:
                         if torch.equal(pos1,torch.zeros(3)):
                              print(i,res)
                              warnings.warn(f"find a atom pos might be unselected in {path}", UserWarning)
                    chi_ang[agi]=_get_torsion(*pos)
          chi_masks.append(chi_mask)
          chi_angs.append(chi_ang)
     return X_ca,X_n,X_c,angls,resid,chi_angs,chi_masks,only_mainchain

def angle_to_sin_cos(degrees,isdegree=True):
    if isdegree:
        radians = torch.deg2rad(degrees).unsqueeze(-1)
    else:
        radians = degrees.unsqueeze(-1)
    return torch.concat([torch.cos(radians.clone()),torch.sin(radians.clone())],dim=-1)

def get_emb_from_pdb(pdb_path,chain = None):
    protokenize = proteintokenizer()
    X_ca,X_n,X_c,angls,resid,chi_angs,chi_masks,only_mainchain = paraser_protein(pdb_path,chain=chain)
    X_ca = np.concatenate(X_ca)
    X_n = np.concatenate(X_n)
    X_c= np.concatenate(X_c)
    angls= np.concatenate(angls)
    resid =np.concatenate(resid)
    chi_angs= np.concatenate(chi_angs)
    chi_masks= np.concatenate(chi_masks)
    only_mainchain = np.array(only_mainchain)
    X_ca = torch.from_numpy(X_ca.reshape(-1,3))
    X_ca = X_ca- X_ca[0]
    X_n = torch.from_numpy(X_n.reshape(-1,3))-X_ca[0]
    X_c = torch.from_numpy(X_c.reshape(-1,3))-X_ca[0]
    angls = torch.from_numpy(angls.reshape(-1,4)[:,:3]) 
    angls =angle_to_sin_cos(angls) 
    resls = [three_to_one[i] for i in resid.reshape(-1,3)[:,1]]
    resls = torch.tensor([protokenize.stoi[i] for i in resls])
    resls[only_mainchain==1]==9
    chi_angs =torch.from_numpy(chi_angs.reshape(-1,4))
    chi_angs[only_mainchain==1]==0
    chi_angs = angle_to_sin_cos(chi_angs,isdegree=False).reshape(-1,8)
    chi_masks =torch.from_numpy(chi_masks.reshape(-1,4,1))
    chi_masks[only_mainchain==1]==0
    chi_masks = torch.concat([chi_masks,chi_masks],dim=-1).reshape(-1,8)
    reslen = X_ca.shape[0]
    return X_ca,X_n,X_c,angls,resls,chi_angs,chi_masks,reslen

def get_encode_from_emb(model,X_ca,X_n,X_c,angls,resls,chi_angs,chi_masks,reslen):
    device = next(model.parameters()).device
    model.eval()
    with torch.set_grad_enabled(False):
        chi_angs = chi_angs.to(device, dtype=torch.float32)
        chi_masks = chi_masks.squeeze().to(device)
        resls = resls.squeeze().to(device)
        angls =angls.to(device, dtype=torch.float32)
        chi_angs=chi_angs*chi_masks.squeeze()
        x_recon,x_seq,x_ang, vq_loss, perplexity, encoding_indices,mask = model(resls.unsqueeze(0),chi_angs,angls.unsqueeze(0),[reslen])
    return x_recon.cpu(),x_seq.cpu(),x_ang.cpu(), vq_loss.cpu(), perplexity.cpu(), encoding_indices.cpu(),mask.cpu()
def get_encoder_from_emb(pdb_path,model,chain = None):
    X_ca,X_n,X_c,angls,resls,chi_angs,chi_masks,reslen= get_emb_from_pdb(pdb_path,chain=chain) 
    x_recon,x_seq,x_ang, vq_loss, perplexity, encoding_indices,mask = get_encode_from_emb(model,X_ca,X_n,X_c,angls,resls,chi_angs,chi_masks,reslen)
    result = {'X_ca':X_ca,
    'X_n':X_n,
    'X_c':X_c,
    'angls':angls,
    'resls':resls,
    'chi_angs':chi_angs,
    'chi_masks':chi_masks,
    'reslen':reslen,
    'x_recon':x_recon,
    'x_seq':x_seq,
    'x_ang':x_ang,
    'vq_loss':vq_loss,
    'perplexity':perplexity,
    'encoding_indices':encoding_indices,
    'mask':mask}
    return result
def get_decode_from_aa_psi(pdb_path,vq,model,chain = None):
     device = next(model.parameters()).device
     X_ca,X_n,X_c,angls,resls,chi_angs,chi_masks,reslen= get_emb_from_pdb(pdb_path,chain=chain)
     with torch.set_grad_enabled(False):
               encodings = torch.zeros(vq.shape[0], model.vq_layer.num_embeddings, device=device)
               encodings.scatter_(1, vq.to(device).unsqueeze(-1), 1)
               quantized = torch.matmul(encodings, model.vq_layer.embeddings.weight) 
               quantized,mask=pads(quantized,[reslen])
               ang,_ = pads(angls.unsqueeze(0).reshape(-1,6).to(device, dtype=torch.float32),[reslen])
               x_recon = model.decoder(quantized+model.tranform(ang),mask)
               x_seq = model.seqhead(x_recon)
               x_ang = model.anghead(x_recon)
     return x_ang.cpu().numpy()
def updata_atom14(atom14,z,atoms,pos):
               t = restype_name_to_atom14_names[resname].index(atoms)
               atoms14[z,t] =pos
def get_pdb_from_backbone(pdb_path,vq,model,chain = None,save_file='./'):
     chi_angs= get_decode_from_aa_psi(pdb_path,vq,model,chain=chain).squeeze()
     u = mda.Universe(pdb_path ,topology_format='PDB')
     chain = chain.item()
     new_dic={chain:[]}
     if chain is not None:
        u = u.select_atoms(f'chainid {chain}')


     for z,res in enumerate(u.residues):

          resname= res.resname
          new_dic[chain].append({resname:{}})
          base_atom = ['N','C','CA','O']
          for i,atoms in enumerate(base_atom):
               new_dic[chain][-1][resname][atoms]=res.atoms.select_atoms(f'name {atoms}').positions[0]
          if resname=='GLY':
               continue
          chi_angle = chi_angles_atoms[resname]
          
          cb = solve_target_vector_with_norm(res.atoms.select_atoms(f'name N').positions[0] ,res.atoms.select_atoms(f'name CA').positions[0] ,res.atoms.select_atoms(f'name C').positions[0],res.atoms.select_atoms(f'name O').positions[0])
          new_positions={atoms:res.atoms.select_atoms(f'name {atoms}').positions[0] for atoms in base_atom}
          new_positions['CB']=cb
          initial_structure={atoms[0]:atoms[2] for atoms in rigid_group_atom_positions[resname]}
          new_pos = align_points(initial_structure, new_positions)
          for ki,k in enumerate(chi_angle):
               a1, a2, a3, a4 = k
               current_angle = _get_torsion(
                    torch.from_numpy(new_pos[a1]).to(torch.float32),
                    torch.from_numpy(new_pos[a2]).to(torch.float32),
                    torch.from_numpy(new_pos[a3]).to(torch.float32),
                    torch.from_numpy(new_pos[a4]).to(torch.float32)
               )%(2 * torch.pi)
               target_angle  = torch.deg2rad(torch.tensor(cos_sin_to_degrees(chi_angs[z,2*ki],chi_angs[z,2*ki+1])))%(2 * torch.pi)
               sub_atom_id = ki+4
               delta_angle = target_angle - current_angle
               delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi
               axis_p1 = new_pos[a2] 
               axis_p2 = new_pos[a3]
               rotated_atoms = [s[0] for s in list(filter(lambda l:l[1]>=sub_atom_id,rigid_group_atom_positions[resname]))]
               for atom in rotated_atoms:
                    new_pos[atom] = rotate_around_axis(
                         points=new_pos[atom],
                         axis_point1=axis_p1,
                         axis_point2=axis_p2,
                         angle=delta_angle
                    )
          for atom,pos in new_pos.items():
          
                    new_dic[chain][-1][resname][atom]=pos
     return new_dic
def get_pdb_from_backbone2(pdb_path,vq,chain = None,save_file='./'):
     chi_angs= get_decode_from_aa_psi(pdb_path,vq,model,chain=chain).squeeze()
     u = mda.Universe(pdb_path ,topology_format='PDB')
     chain = chain.item()
     new_dic={chain:[]}
     if chain is not None:
        u = u.select_atoms(f'chainid {chain}')
     cos_value = chi_angs[:,::2]
     sin_value = chi_angs[:,1::2]
     angsls = np.arctan2(sin_value,cos_value)
     resnames = u.residues.resnames
     base_atom = ['N','C','CA','O']
     new_pos = np.zeros((len(resnames),37,3))
     for i,atoms in enumerate(base_atom):
          new_pos[:,atom_types.index(atoms),:]=u.atoms.select_atoms(f'name {atoms}').positions

     gly_mask = resnames != 'GLY'
     chi_angle2 = [chi_angles_ind_37[resname]  for resname in resnames]
     chi_mask = [chi_angles_mask_dic[resname] for resname in resnames]
     cb = solve_target_vector_with_norm(u.atoms.select_atoms(f'name N').positions ,u.atoms.select_atoms(f'name CA').positions ,u.atoms.select_atoms(f'name C').positions,u.atoms.select_atoms(f'name O').positions)  
     new_pos[:,atom_types.index('CB'),:]=cb*gly_mask[:,None]
     initial_structure=np.stack([rigid_atoms37_position[resname] for resname in resnames],axis=0)
     new_pos = align_points_batch(initial_structure, new_pos)
     for i in range(new_pos.shape[0]):
          current_pos = torch.from_numpy(new_pos[i]).to(torch.float32)
          chi_angles_ind= chi_angle2[i]
          target_angles = torch.from_numpy(angsls[i]).to(torch.float32)
          resname = resnames[i]
     for ki,k in enumerate(chi_angles_ind):
          a1,a2,a3,a4=k
          current_angle = _get_torsion(
                         current_pos[a1],
                         current_pos[a2],
                         current_pos[a3],
                         current_pos[a4]
                    )%(2 * torch.pi)
          target_angle = target_angles[ki]
          sub_atom_id = ki+4
          delta_angle = target_angle - current_angle
          delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi
          axis_p1 = current_pos[a2] 
          axis_p2 = current_pos[a3]
          rotated_atoms = [s[0] for s in list(filter(lambda l:l[1]>=sub_atom_id,rigid_group_atom_positions[resname]))]
          rotated_ind = [atom_types.index(atom) for atom in rotated_atoms]
          current_pos[rotated_ind] = torch.from_numpy(rotate_around_axis(
                              points=current_pos[rotated_ind].numpy(),
                              axis_point1=axis_p1.numpy(),
                              axis_point2=axis_p2.numpy(),
                              angle=delta_angle
                         )).to(torch.float32)
          new_pos[i] = current_pos.numpy()
     return new_pos
def get_model(args):
     model = Con_CustomVQVAE(input_dim=args.n_hiddens, 
                hidden_dim=args.n_hiddens, 
                embedding_dim=args.n_hiddens, 
                num_embeddings=args.n_embeddings,
                ema =args.use_ema,
                commitment_cost=1
                  ).to(args.device)
     return model
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_updates", type=int, default=5000)
    parser.add_argument("--n_hiddens", type=int, default=256)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--use_ema", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--pdb_path", type=str, default=None)
    args = parser.parse_args()
    model = Con_CustomVQVAE(input_dim=args.n_hiddens, 
                hidden_dim=args.n_hiddens, 
                embedding_dim=args.n_hiddens, 
                num_embeddings=args.n_embeddings,
                ema =args.use_ema,
                commitment_cost=1
                  ).to(args.device)
    model.load_state_dict(torch.load('vaemodel2.pt',map_location=args.device))
    model.eval()
    result = get_encoder_from_emb(args.pdb_path,vq,model)
