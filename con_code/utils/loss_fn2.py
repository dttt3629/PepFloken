import torch
import torch.nn as nn
import torch.nn.functional as F

from con_code.common.residue_con import (
    atom_types,
    restype_name_to_atom14_names,
    chi_angles_atoms,
    three_to_one,
)
from pepfloken.modules.protein.constants import AA


def safe_mean(x, mask, eps=1e-8):
    x = x * mask
    return x.sum() / (mask.sum() + eps)


def masked_rmsd_loss(pred, true, mask, eps=1e-8):
    sq = ((pred - true) ** 2).sum(dim=-1)
    return torch.sqrt(safe_mean(sq, mask, eps) + eps)
def smooth_abs(x, beta=0.1):
    return F.smooth_l1_loss(
        x,
        torch.zeros_like(x),
        beta=beta,
        reduction="none",
    )
def masked_smooth_coord_loss(pred, true, mask, beta=0.1, eps=1e-8):
    dist = torch.norm(pred - true, dim=-1)
    loss = F.smooth_l1_loss(
        dist,
        torch.zeros_like(dist),
        beta=beta,
        reduction="none",
    )
    return safe_mean(loss, mask, eps)
def gather_four_atoms(coords, idx):

    gather_idx = idx.unsqueeze(-1).expand(*idx.shape, 3)
    return torch.gather(coords, dim=-2, index=gather_idx)


def dihedral_from_four_points(p0, p1, p2, p3, eps=1e-8):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = b1 / (torch.norm(b1, dim=-1, keepdim=True) + eps)

    v = b0 - (b0 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm
    w = b2 - (b2 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm

    v_norm = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    w_norm = w / (torch.norm(w, dim=-1, keepdim=True) + eps)

    cos_val = (v_norm * w_norm).sum(dim=-1).clamp(-1.0, 1.0)
    sin_val = (torch.cross(b1_norm, v_norm, dim=-1) * w_norm).sum(dim=-1)
    return cos_val, sin_val


def normalize_vec(v, eps=1e-8):
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def build_local_frame_from_backbone(true_atoms, eps=1e-8):

    n = true_atoms[:, :, 0, :]
    ca = true_atoms[:, :, 1, :]
    c = true_atoms[:, :, 2, :]

    e1 = normalize_vec(c - ca, eps=eps)

    u = normalize_vec(n - ca, eps=eps)
    e2 = u - (u * e1).sum(dim=-1, keepdim=True) * e1
    e2 = normalize_vec(e2, eps=eps)

    e3 = torch.cross(e1, e2, dim=-1)
    e3 = normalize_vec(e3, eps=eps)

    basis = torch.stack([e1, e2, e3], dim=-2)  
    center = ca
    return basis, center


def global_to_local_points(points, basis, center):

    vec = points - center.unsqueeze(-2)  
    local = torch.einsum("blij,blaj->blai", basis, vec)
    return local
def angle_cos_from_three_points(p0, p1, p2, eps=1e-8):

    v1 = p0 - p1
    v2 = p2 - p1

    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)

    return (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)

def gather_neighbors_from_residue_dim(x, neighbor_idx):

    B, L, K = neighbor_idx.shape
    device = neighbor_idx.device

    offset = (torch.arange(B, device=device) * L).view(B, 1, 1)   
    flat_idx = (neighbor_idx + offset).reshape(-1)                

    if x.dim() == 4:
        _, _, A, C = x.shape
        flat_x = x.reshape(B * L, A, C)                           
        out = flat_x[flat_idx].reshape(B, L, K, A, C)             
        return out

    elif x.dim() == 3:
        _, _, A = x.shape
        flat_x = x.reshape(B * L, A)                              
        out = flat_x[flat_idx].reshape(B, L, K, A)                
        return out

    else:
        raise ValueError(
            f"Unsupported tensor rank for gather_neighbors_from_residue_dim: {x.dim()}"
        )

RESTYPE_ORDER = {aa.name: aa.value for aa in AA}
RESTYPE_ORDER['PAD']=21
RESTYPES = list(RESTYPE_ORDER.keys()    )
SIDECHAIN_BONDS = {
    "ALA": [("CA", "CB")],
    "ARG": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    "ASN": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "ASP": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "CYS": [("CA", "CB"), ("CB", "SG")],
    "GLN": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    "GLU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    "GLY": [],
    "HIS": [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CD2", "NE2")],
    "ILE": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
    "LEU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "LYS": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    "MET": [("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "PHE": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")],
    "PRO": [("CA", "CB"), ("CB", "CG"), ("CG", "CD")],
    "SER": [("CA", "CB"), ("CB", "OG")],
    "THR": [("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
    "TRP": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("CD2", "CE2"),
            ("CD2", "CE3"), ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2")],
    "TYR": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CZ", "OH")],
    "VAL": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
    "UNK": [("CA", "CB")],
}

BACKBONE_ANCHOR_BONDS = [
    ("N", "CA"),
    ("CA", "C"),
    ("C", "O"),
]
def build_restype_atom14_name_map():
    atom_name_to_idx = {}
    for resname, atom_names in restype_name_to_atom14_names.items():
        d = {}
        for i, atom_name in enumerate(atom_names):
            if atom_name != "":
                d[atom_name] = i
        atom_name_to_idx[resname] = d
    return atom_name_to_idx


def build_chi_atom_index_table():
    atom_name_to_idx = build_restype_atom14_name_map()

    chi_atom_indices = torch.zeros((len(RESTYPES), 4, 4), dtype=torch.long)
    chi_defined_mask = torch.zeros((len(RESTYPES), 4), dtype=torch.float32)

    for resname, chis in chi_angles_atoms.items():
        if resname not in RESTYPE_ORDER:
            continue
        r = RESTYPE_ORDER[resname]
        for chi_i, atom_names in enumerate(chis):
            ok = True
            idxs = []
            for atom_name in atom_names:
                if atom_name not in atom_name_to_idx[resname]:
                    ok = False
                    break
                idxs.append(atom_name_to_idx[resname][atom_name])
            if ok:
                chi_atom_indices[r, chi_i] = torch.tensor(idxs, dtype=torch.long)
                chi_defined_mask[r, chi_i] = 1.0

    return chi_atom_indices, chi_defined_mask


def build_sidechain_bond_list_table():
    max_bonds = 10
    bond_idx = torch.zeros((len(RESTYPES), max_bonds, 2), dtype=torch.long)
    bond_mask = torch.zeros((len(RESTYPES), max_bonds), dtype=torch.float32)

    atom_name_to_idx = build_restype_atom14_name_map()

    sidechain_bonds = {
        "ALA": [("CA", "CB")],
        "ARG": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
        "ASN": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
        "ASP": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
        "CYS": [("CA", "CB"), ("CB", "SG")],
        "GLN": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
        "GLU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
        "GLY": [],
        "HIS": [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CD2", "NE2")],
        "ILE": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
        "LEU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
        "LYS": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
        "MET": [("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
        "PHE": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")],
        "PRO": [("CA", "CB"), ("CB", "CG"), ("CG", "CD")],
        "SER": [("CA", "CB"), ("CB", "OG")],
        "THR": [("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
        "TRP": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("CD2", "CE2"),
                ("CD2", "CE3"), ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2")],
        "TYR": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CZ", "OH")],
        "VAL": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
        "UNK": [("CA", "CB")],
    }

    for resname in RESTYPES:
        if resname not in RESTYPE_ORDER:
            continue
        r = RESTYPE_ORDER[resname]
        valid_bonds = []
        for a, b in sidechain_bonds.get(resname, []):
            if a in atom_name_to_idx[resname] and b in atom_name_to_idx[resname]:
                valid_bonds.append((atom_name_to_idx[resname][a], atom_name_to_idx[resname][b]))

        for i, (a, b) in enumerate(valid_bonds[:max_bonds]):
            bond_idx[r, i, 0] = a
            bond_idx[r, i, 1] = b
            bond_mask[r, i] = 1.0

    return bond_idx, bond_mask

def build_sidechain_angle_list_table():
    max_angles = 32

    num_restypes = max(RESTYPE_ORDER.values()) + 1
    angle_idx = torch.zeros((num_restypes, max_angles, 3), dtype=torch.long)
    angle_mask = torch.zeros((num_restypes, max_angles), dtype=torch.float32)

    atom_name_to_idx = build_restype_atom14_name_map()

    for resname in RESTYPES:
        if resname not in RESTYPE_ORDER:
            continue
        if resname not in atom_name_to_idx:
            continue

        r = RESTYPE_ORDER[resname]
        name_to_idx = atom_name_to_idx[resname]

        bonds = BACKBONE_ANCHOR_BONDS + SIDECHAIN_BONDS.get(resname, [])

        adj = {}
        for a_name, b_name in bonds:
            if a_name not in name_to_idx or b_name not in name_to_idx:
                continue

            a = name_to_idx[a_name]
            b = name_to_idx[b_name]

            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        angles = []

        for center_atom, neighbors in adj.items():
            neighbors = sorted(list(neighbors))

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a = neighbors[i]
                    c = neighbors[j]
                    b = center_atom

                    if a >= 4 or b >= 4 or c >= 4:
                        angles.append((a, b, c))

        for i, (a, b, c) in enumerate(angles[:max_angles]):
            angle_idx[r, i, 0] = a
            angle_idx[r, i, 1] = b
            angle_idx[r, i, 2] = c
            angle_mask[r, i] = 1.0

    return angle_idx, angle_mask
class LocalSidechainLossForGenModel(nn.Module):

    def __init__(
        self,
        config, 
    ):
        super().__init__()

        self.include_cb = config.include_cb
        self.target_coord_weight = config.target_coord_weight
        self.pair_distance_weight = config.pair_distance_weight
        self.chi_angle_weight = config.chi_angle_weight
        self.bond_length_weight = config.bond_length_weight
        self.env_interaction_weight = config.env_interaction_weight
        self.num_env_neighbors = config.num_env_neighbors
        self.env_use_sidechain_only = config.env_use_sidechain_only

        chi_atom_indices, chi_defined_mask = build_chi_atom_index_table()
        bond_idx_table, bond_mask_table = build_sidechain_bond_list_table()
        self.bond_angle_weight = config.bond_angle_weight

        angle_idx_table, angle_mask_table = build_sidechain_angle_list_table()

        self.register_buffer("angle_idx_table", angle_idx_table)
        self.register_buffer("angle_mask_table", angle_mask_table)
        self.register_buffer("chi_atom_indices", chi_atom_indices)
        self.register_buffer("chi_defined_mask", chi_defined_mask)
        self.register_buffer("bond_idx_table", bond_idx_table)
        self.register_buffer("bond_mask_table", bond_mask_table)

    @staticmethod
    def make_sidechain_mask(atoms_mask, include_cb=True, include_oxt=False):

        A = atoms_mask.shape[-1]
        sc_mask = atoms_mask.clone().float()

        for idx in [0, 1, 2, 3]:
            if idx < A:
                sc_mask[..., idx] = 0.0

        if (not include_cb) and A > 4:
            sc_mask[..., 4] = 0.0

        if (A == 15) and (not include_oxt):
            sc_mask[..., 14] = 0.0

        return sc_mask

    def build_residue_sidechain_mask(self, atoms_mask, res_mask, gen_mask):

        center_mask = (res_mask.bool() & gen_mask.bool()).float()  
        sc_mask = self.make_sidechain_mask(
            atoms_mask,
            include_cb=self.include_cb,
            include_oxt=False,
        )
        sc_mask = sc_mask * center_mask[..., None]
        return sc_mask, center_mask
    def bond_angle_loss_fn(
        self,
        pred_local_atoms,
        true_local_atoms,
        atoms_mask,
        aa,
        res_mask,
        gen_mask,
    ):

        pred14 = pred_local_atoms[..., :14, :]
        true14 = true_local_atoms[..., :14, :]
        mask14 = atoms_mask[..., :14].float()

        center_mask = (res_mask.bool() & gen_mask.bool()).float()
        mask14 = mask14 * center_mask[..., None]

        per_res_angle_idx = self.angle_idx_table[aa]       
        per_res_angle_mask = self.angle_mask_table[aa]     

        flat_pred = pred14.reshape(-1, 14, 3)
        flat_true = true14.reshape(-1, 14, 3)
        flat_mask = mask14.reshape(-1, 14)

        flat_angle_idx = per_res_angle_idx.reshape(
            -1, per_res_angle_idx.shape[-2], 3
        )
        flat_angle_mask = per_res_angle_mask.reshape(
            -1, per_res_angle_mask.shape[-1]
        )

        flat_center = center_mask.reshape(-1, 1)

        _, max_angles = flat_angle_mask.shape

        losses = []
        masks = []

        for a_i in range(max_angles):
            idx0 = flat_angle_idx[:, a_i, 0]
            idx1 = flat_angle_idx[:, a_i, 1]
            idx2 = flat_angle_idx[:, a_i, 2]

            p0_pred = torch.gather(
                flat_pred, 1, idx0[:, None, None].expand(-1, 1, 3)
            ).squeeze(1)
            p1_pred = torch.gather(
                flat_pred, 1, idx1[:, None, None].expand(-1, 1, 3)
            ).squeeze(1)
            p2_pred = torch.gather(
                flat_pred, 1, idx2[:, None, None].expand(-1, 1, 3)
            ).squeeze(1)

            p0_true = torch.gather(
                flat_true, 1, idx0[:, None, None].expand(-1, 1, 3)
            ).squeeze(1)
            p1_true = torch.gather(
                flat_true, 1, idx1[:, None, None].expand(-1, 1, 3)
            ).squeeze(1)
            p2_true = torch.gather(
                flat_true, 1, idx2[:, None, None].expand(-1, 1, 3)
            ).squeeze(1)

            pred_cos = angle_cos_from_three_points(p0_pred, p1_pred, p2_pred)
            true_cos = angle_cos_from_three_points(p0_true, p1_true, p2_true)

            loss = (pred_cos - true_cos) ** 2

            valid = (
                flat_angle_mask[:, a_i]
                * torch.gather(flat_mask, 1, idx0[:, None]).squeeze(1)
                * torch.gather(flat_mask, 1, idx1[:, None]).squeeze(1)
                * torch.gather(flat_mask, 1, idx2[:, None]).squeeze(1)
                * flat_center.squeeze(-1)
            )

            losses.append(loss)
            masks.append(valid)

        losses = torch.stack(losses, dim=-1)
        masks = torch.stack(masks, dim=-1)

        return safe_mean(losses, masks)
    def target_coord_loss_fn(
        self,
        pred_local_atoms,
        true_local_atoms,
        atoms_mask,
        res_mask,
        gen_mask,
    ):
        sc_mask, _ = self.build_residue_sidechain_mask(atoms_mask, res_mask, gen_mask)
        return 0.1 * masked_rmsd_loss(pred_local_atoms, true_local_atoms, sc_mask)+1.0*masked_smooth_coord_loss(
                pred_local_atoms,
                true_local_atoms,
                sc_mask,
                beta=0.1,
            )

    def pair_distance_loss_fn(
        self,
        pred_local_atoms,
        true_local_atoms,
        atoms_mask,
        res_mask,
        gen_mask,
    ):

        sc_mask, center_mask = self.build_residue_sidechain_mask(
            atoms_mask, res_mask, gen_mask
        )

        B, L, A, _ = pred_local_atoms.shape

        anchor_mask = torch.zeros_like(sc_mask).float()

        for idx in [0, 1, 2]:
            if idx < A:
                anchor_mask[..., idx] = atoms_mask[..., idx].float()

        anchor_mask = anchor_mask * center_mask[..., None]

        geom_mask = torch.clamp(sc_mask + anchor_mask, max=1.0)

        pred_d = torch.cdist(pred_local_atoms, pred_local_atoms)   
        true_d = torch.cdist(true_local_atoms, true_local_atoms)   

        pair_mask = geom_mask.unsqueeze(-1) * geom_mask.unsqueeze(-2)

        eye = torch.eye(A, device=pred_local_atoms.device).view(1, 1, A, A)
        pair_mask = pair_mask * (1.0 - eye)

        anchor_pair_mask = anchor_mask.unsqueeze(-1) * anchor_mask.unsqueeze(-2)
        pair_mask = pair_mask * (1.0 - anchor_pair_mask)

        diff = smooth_abs(pred_d - true_d, beta=0.1)

        return safe_mean(diff, pair_mask)
    def chi_angle_loss_fn(
        self,
        pred_local_atoms,
        aa,
        chi_angs,
        chi_masks,
        atoms_mask,
        res_mask,
        gen_mask,
    ):

        pred_atom14 = pred_local_atoms[..., :14, :]
        mask14 = atoms_mask[..., :14].float()

        center_mask = (res_mask.bool() & gen_mask.bool()).float()   
        mask14 = mask14 * center_mask[..., None]
        chi_angs =torch.stack([torch.cos(chi_angs), torch.sin(chi_angs)], dim=-1)
        per_res_chi_idx = self.chi_atom_indices[aa]      
        per_res_chi_def = self.chi_defined_mask[aa]      

        flat_pred = pred_atom14.reshape(-1, 14, 3)
        flat_idx = per_res_chi_idx.reshape(-1, 4, 4)
        flat_mask14 = mask14.reshape(-1, 14)
        flat_chi_gt = chi_angs.reshape(-1, 4, 2)
        flat_chi_mask = chi_masks.reshape(-1, 4)
        flat_chi_def = per_res_chi_def.reshape(-1, 4)
        flat_center = center_mask.reshape(-1, 1)         

        pred_chi_cos = []
        pred_chi_sin = []
        pred_valid = []

        for chi_i in range(4):
            idx4 = flat_idx[:, chi_i, :]                 
            atoms4 = gather_four_atoms(flat_pred, idx4)  

            p0, p1, p2, p3 = atoms4[:, 0], atoms4[:, 1], atoms4[:, 2], atoms4[:, 3]
            cos_val, sin_val = dihedral_from_four_points(p0, p1, p2, p3)

            atom_valid = torch.gather(flat_mask14, dim=-1, index=idx4).prod(dim=-1)
            valid = atom_valid * flat_chi_mask[:, chi_i] * flat_chi_def[:, chi_i] * flat_center.squeeze(-1)

            pred_chi_cos.append(cos_val)
            pred_chi_sin.append(sin_val)
            pred_valid.append(valid)

        pred_chi_cos = torch.stack(pred_chi_cos, dim=-1)
        pred_chi_sin = torch.stack(pred_chi_sin, dim=-1)
        pred_valid = torch.stack(pred_valid, dim=-1)

        gt_cos = flat_chi_gt[..., 0]
        gt_sin = flat_chi_gt[..., 1]

        chi_err = (pred_chi_cos - gt_cos) ** 2 + (pred_chi_sin - gt_sin) ** 2
        return safe_mean(chi_err, pred_valid)

    def bond_length_loss_fn(
        self,
        pred_local_atoms,
        true_local_atoms,
        atoms_mask,
        aa,
        res_mask,
        gen_mask,
    ):
        pred14 = pred_local_atoms[..., :14, :]
        true14 = true_local_atoms[..., :14, :]
        mask14 = atoms_mask[..., :14].float()

        center_mask = (res_mask.bool() & gen_mask.bool()).float()
        mask14 = mask14 * center_mask[..., None]

        per_res_bond_idx = self.bond_idx_table[aa]       
        per_res_bond_mask = self.bond_mask_table[aa]     

        flat_pred = pred14.reshape(-1, 14, 3)
        flat_true = true14.reshape(-1, 14, 3)
        flat_mask = mask14.reshape(-1, 14)
        flat_bond_idx = per_res_bond_idx.reshape(-1, per_res_bond_idx.shape[-2], 2)
        flat_bond_mask = per_res_bond_mask.reshape(-1, per_res_bond_mask.shape[-1])
        flat_center = center_mask.reshape(-1, 1)

        _, max_bonds = flat_bond_mask.shape
        losses = []
        masks = []

        for b in range(max_bonds):
            idx_a = flat_bond_idx[:, b, 0]
            idx_c = flat_bond_idx[:, b, 1]

            pa_pred = torch.gather(flat_pred, 1, idx_a[:, None, None].expand(-1, 1, 3)).squeeze(1)
            pc_pred = torch.gather(flat_pred, 1, idx_c[:, None, None].expand(-1, 1, 3)).squeeze(1)

            pa_true = torch.gather(flat_true, 1, idx_a[:, None, None].expand(-1, 1, 3)).squeeze(1)
            pc_true = torch.gather(flat_true, 1, idx_c[:, None, None].expand(-1, 1, 3)).squeeze(1)

            pred_len = torch.norm(pa_pred - pc_pred, dim=-1)
            true_len = torch.norm(pa_true - pc_true, dim=-1)

            valid = (
                flat_bond_mask[:, b]
                * torch.gather(flat_mask, 1, idx_a[:, None]).squeeze(1)
                * torch.gather(flat_mask, 1, idx_c[:, None]).squeeze(1)
                * flat_center.squeeze(-1)
            )

            losses.append(smooth_abs(pred_len - true_len, beta=0.05))
            masks.append(valid)

        losses = torch.stack(losses, dim=-1)
        masks = torch.stack(masks, dim=-1)

        return safe_mean(losses, masks)

    def env_interaction_loss_fn(
        self,
        pred_local_atoms,
        true_local_atoms,
        true_atoms,
        atoms_mask,
        res_mask,
        gen_mask,
        rot=None,
        trans=None,
    ):

        device = pred_local_atoms.device
        B, L, A, _ = pred_local_atoms.shape
        K = min(self.num_env_neighbors, L)

        center_res_mask = (res_mask.bool() & gen_mask.bool())      
        env_res_mask = (res_mask.bool() & (~gen_mask.bool()))      

        center_sc_mask = self.make_sidechain_mask(
            atoms_mask,
            include_cb=self.include_cb,
            include_oxt=False,
        ).float()                                                  
        center_sc_mask = center_sc_mask * center_res_mask[..., None].float()

        if self.env_use_sidechain_only:
            env_atom_mask = self.make_sidechain_mask(
                atoms_mask,
                include_cb=True,
                include_oxt=False,
            ).float()
        else:
            env_atom_mask = atoms_mask.float()

        env_atom_mask = env_atom_mask * env_res_mask[..., None].float()   

        ca = true_atoms[:, :, 1, :]                                       
        ca_dist = torch.cdist(ca, ca)                                     

        valid_neighbor_mask = env_res_mask[:, None, :].expand(B, L, L)    
        big_val = torch.full_like(ca_dist, 1e6)
        masked_dist = torch.where(valid_neighbor_mask, ca_dist, big_val)

        _, neighbor_idx = torch.topk(
            masked_dist, k=K, dim=-1, largest=False
        )                                                                  

        neighbor_valid = torch.gather(
            valid_neighbor_mask.float(), dim=-1, index=neighbor_idx
        )                                                                  

        env_true_global = gather_neighbors_from_residue_dim(true_atoms, neighbor_idx)    
        env_atom_mask_k = gather_neighbors_from_residue_dim(env_atom_mask, neighbor_idx)  

        if rot is None or trans is None:
            rot, trans = build_local_frame_from_backbone(true_atoms)  

        rot_expand = rot.unsqueeze(2).expand(B, L, K, 3, 3)          
        trans_expand = trans.unsqueeze(2).expand(B, L, K, 3)         

        env_vec = env_true_global - trans_expand.unsqueeze(-2)       
        env_true_local_in_center = torch.einsum(
            "blkij,blkaj->blkai", rot_expand, env_vec
        )                                                             

        pred_center = pred_local_atoms.unsqueeze(2)                   
        true_center = true_local_atoms.unsqueeze(2)                   

        pred_d = torch.cdist(
            pred_center.reshape(B * L, A, 3),
            env_true_local_in_center.reshape(B * L, K * A, 3),
        ).reshape(B, L, A, K, A).permute(0, 1, 3, 2, 4)             

        true_d = torch.cdist(
            true_center.reshape(B * L, A, 3),
            env_true_local_in_center.reshape(B * L, K * A, 3),
        ).reshape(B, L, A, K, A).permute(0, 1, 3, 2, 4)             

        pair_mask = (
            center_sc_mask.unsqueeze(2).unsqueeze(-1) *              
            env_atom_mask_k.unsqueeze(-2) *                          
            neighbor_valid.unsqueeze(-1).unsqueeze(-1) *             
            center_res_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        )                                                            

        diff = torch.abs(pred_d - true_d)
        return safe_mean(diff, pair_mask)

    def forward(
        self,
        pred_global_atoms,
        true_atoms,
        atoms_mask,
        aa,
        chi_angs,
        chi_masks,
        res_mask,
        gen_mask,
    ):

        basis, center = build_local_frame_from_backbone(true_atoms)
        true_local_atoms = global_to_local_points(true_atoms, basis, center)
        pred_local_atoms = global_to_local_points(pred_global_atoms, basis, center)
        pred_for_geom = pred_local_atoms.clone()
        pred_for_geom[..., 0:4, :] = true_local_atoms[..., 0:4, :]

        l_target = self.target_coord_loss_fn(
            pred_for_geom,
            true_local_atoms,
            atoms_mask,
            res_mask,
            gen_mask,
        )

        l_pair = self.pair_distance_loss_fn(
            pred_for_geom,
            true_local_atoms,
            atoms_mask,
            res_mask,
            gen_mask,
        )

        l_chi = self.chi_angle_loss_fn(
            pred_for_geom,
            aa,
            chi_angs,
            chi_masks,
            atoms_mask,
            res_mask,
            gen_mask,
        )

        l_bond = self.bond_length_loss_fn(
            pred_for_geom,
            true_local_atoms,
            atoms_mask,
            aa,
            res_mask,
            gen_mask,
        )
        l_angle = self.bond_angle_loss_fn(
        pred_for_geom,
        true_local_atoms,
        atoms_mask,
        aa,
        res_mask,
        gen_mask,
            )
        l_env = self.env_interaction_loss_fn(
            pred_local_atoms=pred_for_geom,
            true_local_atoms=true_local_atoms,
            true_atoms=true_atoms,
            atoms_mask=atoms_mask,
            res_mask=res_mask,
            gen_mask=gen_mask,
            rot=basis,
            trans=center,
        )

        total = (
            self.target_coord_weight * l_target
            + self.pair_distance_weight * l_pair
            + self.chi_angle_weight * l_chi
            + self.bond_length_weight * l_bond
            + self.bond_angle_weight * l_angle
            + self.env_interaction_weight * l_env
        )

        center_mask = (res_mask.bool() & gen_mask.bool()).float()
        env_mask = (res_mask.bool() & (~gen_mask.bool())).float()

        return total, {
            "target_coord_loss": l_target,
            "pair_distance_loss": l_pair,
            "chi_angle_loss": l_chi,
            "bond_length_loss": l_bond,
            "bond_angle_loss": l_angle,
            "env_interaction_loss": l_env,
            "total_loss": total,
            "num_gen_res": center_mask.sum().detach(),
            "num_env_res": env_mask.sum().detach(),
        }