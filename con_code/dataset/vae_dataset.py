from torch.utils.data import Dataset
import pickle 
import math
import numpy as np
from con_code.common.residue_con import chi_angles_atoms,three_to_one,resatoms 
import torch
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
one_to_three ={q:k for k,q in three_to_one.items()}
def angle_to_sin_cos(degrees,isdegree=True):
    if isdegree:
        radians = torch.deg2rad(degrees).unsqueeze(-1)
    else:
        radians = degrees.unsqueeze(-1)
    return torch.concat([torch.cos(radians.clone()),torch.sin(radians.clone())],dim=-1)
def batch_global_to_local_14(
    batch_coords: np.ndarray,  
    n_idx: int = 0,            
    ca_idx: int = 1,           
    c_idx: int = 2             
) -> tuple[np.ndarray, np.ndarray]:

    if batch_coords.ndim != 3 or batch_coords.shape[1:] != (15, 3):
        raise ValueError(f"输入必须是 [B, 15, 3] 格式，当前shape: {batch_coords.shape}")

    B = batch_coords.shape[0]  

    N = batch_coords[:, n_idx, :]    
    CA = batch_coords[:, ca_idx, :]  
    C = batch_coords[:, c_idx, :]    

    x_axis = N - CA  
    x_norm = np.linalg.norm(x_axis, axis=1, keepdims=True)  
    x_axis = x_axis / np.where(x_norm > 1e-8, x_norm, 1.0)  

    c_vec = C - CA  
    z_axis = np.cross(x_axis, c_vec)  
    z_norm = np.linalg.norm(z_axis, axis=1, keepdims=True)
    z_axis = z_axis / np.where(z_norm > 1e-8, z_norm, 1.0)

    y_axis = np.cross(z_axis, x_axis)  
    y_norm = np.linalg.norm(y_axis, axis=1, keepdims=True)
    y_axis = y_axis / np.where(y_norm > 1e-8, y_norm, 1.0)

    batch_rot_mats = np.stack([x_axis, y_axis, z_axis], axis=2)  

    translated = batch_coords - CA[:, np.newaxis, :]  

    batch_local_coords = np.matmul(batch_rot_mats.transpose(0, 2, 1), translated.transpose(0, 2, 1))
    batch_local_coords = batch_local_coords.transpose(0, 2, 1)

    ca_local = batch_local_coords[:, ca_idx, :]  
    max_ca_error = np.max(np.abs(ca_local))
    if max_ca_error > 1e-6:
        print(f"警告：CA原子局部坐标偏差过大（最大{max_ca_error:.6f}）")

    return batch_local_coords, batch_rot_mats
class vae_dataset(Dataset):

    def __init__(self,paths):
        super().__init__()
        self.crop = 768
        self.paths = paths
        AA_set = sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','-','!','&','?','|','b'])
        self.stoi = {AA_set[i]: i for i in range(len(AA_set))} 
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        with open(f'{self.paths[index]}','rb') as f:
            X_ca,X_n,X_c,angls,resid,chi_angs,chi_masks,only_mainchain,atoms14,atoms14_mask = pickle.load(f)
            X_ca = np.concatenate(X_ca)
            X_n = np.concatenate(X_n)
            X_c= np.concatenate(X_c)
            angls= np.concatenate(angls)
            resid =np.concatenate(resid)
            chi_angs= np.concatenate(chi_angs)
            chi_masks= np.concatenate(chi_masks)
            only_mainchain = np.array(only_mainchain)
            atoms14 = np.array(atoms14)
            atoms14_mask = np.array(atoms14_mask)

        local_atoms14,batch_rot_mats = batch_global_to_local_14(atoms14)
        atoms14 = torch.from_numpy(atoms14)
        local_atoms14 = torch.from_numpy(local_atoms14)
        batch_rot_mats = torch.from_numpy(batch_rot_mats)
        atoms14_mask = torch.from_numpy(atoms14_mask)
        atoms14_mask = atoms14_mask.reshape(-1,15)

        X_ca = torch.from_numpy(X_ca.reshape(-1,3))
        X_ca = X_ca- X_ca[0]
        X_n = torch.from_numpy(X_n.reshape(-1,3))-X_ca[0]
        X_c = torch.from_numpy(X_c.reshape(-1,3))-X_ca[0]
        angls = torch.from_numpy(angls.reshape(-1,4)[:,:3]) 
        angls =angle_to_sin_cos(angls) 
        resls = [three_to_one[i] for i in resid.reshape(-1,3)[:,1]]
        resls = torch.tensor([self.stoi[i] for i in resls])
        resls[only_mainchain==1]==9
        chi_angs =torch.from_numpy(chi_angs.reshape(-1,4))
        chi_angs[only_mainchain==1]==0
        chi_angs = angle_to_sin_cos(chi_angs,isdegree=False).reshape(-1,8)
        chi_masks =torch.from_numpy(chi_masks.reshape(-1,4,1))
        chi_masks[only_mainchain==1]==0
        chi_masks = torch.concat([chi_masks,chi_masks],dim=-1).reshape(-1,8)
        reslen = X_ca.shape[0]
        if reslen>self.crop:
            X_ca=X_ca[:self.crop]
            X_n=X_n[:self.crop]
            X_c = X_c[:self.crop]
            angls =angls[:self.crop]
            resls =resls [:self.crop]
            chi_angs =chi_angs[:self.crop]
            chi_masks = chi_masks[:self.crop]
            local_atoms14 = local_atoms14[:self.crop]
            batch_rot_mats = batch_rot_mats[:self.crop]
            atoms14_mask = atoms14_mask[:self.crop]
            atoms14 = atoms14[:self.crop]
            reslen = self.crop

        return self.paths[index],X_ca,X_n,X_c,angls,resls,chi_angs,chi_masks,reslen,local_atoms14,atoms14,batch_rot_mats,atoms14_mask
def pep_collate_fn(batch):
    proteinpath,X_ca,X_n,X_c,angls,resls,chi_angs,chi_masks,reslen,local_atoms14,atoms14,batch_rot_mats,atoms14_mask = zip(*batch)      
    X_ca = torch.concat(X_ca)
    X_n = torch.concat(X_n)
    X_c = torch.concat(X_c)
    angls = torch.concat(angls)
    chi_angs = torch.concat(chi_angs)
    chi_masks = torch.concat(chi_masks)
    resls = torch.concat(resls)
    local_atoms14 = torch.concat(local_atoms14)
    batch_rot_mats = torch.concat(batch_rot_mats)
    atoms14_mask = torch.concat(atoms14_mask)
    atoms14 = torch.concat(atoms14)

    return proteinpath,X_ca,X_n,X_c,angls,resls,chi_angs,chi_masks,reslen,local_atoms14,atoms14,batch_rot_mats,atoms14_mask     


class sgvae_dataset(Dataset):

    def __init__(self,datals,dataind):

        super().__init__()
        self.crop = 768
        self.resls = []
        self.ang = []
        for k in datals:
            for i in dataind[k]:
                self.resls.append(k)
                self.ang.append(datals[k][i])
        AA_set = sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','-','!','&','?','|','b'])
        self.stoi = {AA_set[i]: i for i in range(len(AA_set))} 
    def __len__(self):
        return len(self.resls)
    def __getitem__(self, index):
        res = self.resls[index]
        mask =torch.zeros(1,8)
        mask[:,:len(chi_angles_atoms[one_to_three[res]])*2]=1
        resls= torch.Tensor([self.stoi[res]]).to(torch.long)
        return resls,torch.tensor(self.ang[index]),mask

