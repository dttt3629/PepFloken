import torch
import torch.nn.functional as F
from pepfloken.modules.protein.constants import BBHeavyAtom
from pepfloken.data import all_atom


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = t[..., 0]
    y = t[..., 1]
    z = t[..., 2]
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )

def invert_rot_mat(rot_mat: torch.Tensor) -> torch.Tensor:
    return torch.transpose(rot_mat, dim0=-2, dim1=-1)

def global_to_local(
    global_pts: torch.Tensor,
    trans: torch.Tensor,
    rot_mats: torch.Tensor
) -> torch.Tensor:
    global_vec = global_pts - trans
    inv_rot_mats = invert_rot_mat(rot_mats)
    local_pts = rot_vec_mul(inv_rot_mats, global_vec)
    return local_pts

def local_to_global(
    local_pts: torch.Tensor,
    trans: torch.Tensor,
    rot_mats: torch.Tensor
) -> torch.Tensor:
    global_vec = rot_vec_mul(rot_mats, local_pts)
    global_pts = global_vec + trans
    return global_pts

def trans_l2_loss(pred_trans,gen_mask):
    std_dis = 3.8
    loss = (torch.norm(pred_trans[:,:-1,:]-pred_trans[:,1:,:], dim=-1) - std_dis)**2
    loss= loss * (gen_mask[:, :-1] * gen_mask[:, 1:])
    loss = loss.sum(-1)/(gen_mask[:, :-1] * gen_mask[:, 1:]).sum(-1)
    return loss.mean()

def batch_calculate_angles_torch(points_A,points_B,points_C):
    vec_BA = points_A - points_B
    vec_BC = points_C - points_B
    norm_BA = torch.norm(vec_BA, dim=-1) + 1e-8
    norm_BC = torch.norm(vec_BC, dim=-1) + 1e-8
    dot_product = torch.sum(vec_BA * vec_BC, dim=-1)
    cos_theta = dot_product / (norm_BA * norm_BC)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-8, 1.0 - 1e-8)
    theta_rad = torch.acos(cos_theta)
    return theta_rad

def get_pep_bond_metric(pos):
    CA = pos[:, :, BBHeavyAtom.CA]
    N = pos[:, :, BBHeavyAtom.N]
    C = pos[:, :, BBHeavyAtom.C]
    C_bond_N = torch.norm(C[:,:-1,:]-N[:,1:,:], dim=-1)
    CA_C_N = batch_calculate_angles_torch(CA[:,:-1],C[:,:-1],N[:,1:])
    C_N_CA = batch_calculate_angles_torch(C[:,:-1],N[:,1:],CA[:,1:])
    return C_bond_N,CA_C_N,C_N_CA

def pep_bond_loss(full_pred_atoms,full_gt_atoms,gen_mask,weights_a=1,weights_b=0.5,weights_c=0.5):
    pred_C_bond_N,pred_CA_C_N,pred_C_N_CA = get_pep_bond_metric(full_pred_atoms)
    C_bond_N,CA_C_N,C_N_CA = get_pep_bond_metric(full_gt_atoms)
    valid_mask = gen_mask[:, 1:] * gen_mask[:, :-1]
    bond_length_loss = (pred_C_bond_N - C_bond_N)**2 * valid_mask
    ca_c_n_loss = (pred_CA_C_N - CA_C_N)**2 * valid_mask
    c_n_ca_loss = (pred_C_N_CA - C_N_CA)**2 * valid_mask
    loss =  weights_a * bond_length_loss + weights_b * ca_c_n_loss + weights_c * c_n_ca_loss
    loss = loss.sum(-1)/valid_mask.sum(-1)
    return loss.mean()

def rec_loss(full_pred_atoms,full_gt_atoms,gen_mask,res_mask):
    heavy_atoms = [BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.N, BBHeavyAtom.O]
    pred_atoms = full_pred_atoms[:, :, heavy_atoms]
    preddis = torch.norm(pred_atoms.unsqueeze(2)-pred_atoms.unsqueeze(1), dim=-1)
    gt_atoms = full_gt_atoms[:, :, heavy_atoms]
    gdis = torch.norm(gt_atoms.unsqueeze(2)-gt_atoms.unsqueeze(1), dim=-1)
    loss = (preddis - gdis)**2 * ((gen_mask.unsqueeze(-1)*res_mask.unsqueeze(-2)).unsqueeze(-1))
    eps = 1e-8
    valid_pairs_per_sample = (gen_mask.sum(1) * res_mask.sum(1)).clamp(min=eps)
    B = full_pred_atoms.shape[0]
    denominator = (B * valid_pairs_per_sample).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    loss = loss / denominator
    return loss.sum()

def time_dependent_vq_loss(
    vq_logits,
    target_vq,
    gen_mask,
    t,
    vq_size = 1024,
    eps_min = 0.02,
    eps_max = 0.30,
    mode = "same_aa",
    aa_target = None,
    vq_to_seq_matrix = None,
    reduce = True,
):
    assert vq_logits.dim() == 3, vq_logits.shape
    assert target_vq.shape == vq_logits.shape[:2], (target_vq.shape, vq_logits.shape)
    assert gen_mask.shape == target_vq.shape, (gen_mask.shape, target_vq.shape)

    B, L, K = vq_logits.shape
    assert K == vq_size, (K, vq_size)

    device = vq_logits.device
    dtype = vq_logits.dtype

    target_vq = torch.clamp(target_vq.long(), 0, K - 1)
    mask = gen_mask.float()

    if t.dim() == 2:
        t_scalar = t.squeeze(-1)
    else:
        t_scalar = t
    t_scalar = t_scalar.detach().clamp(0.0, 1.0)

    eps = eps_min + (eps_max - eps_min) * (1.0 - t_scalar)
    eps = eps.to(device=device, dtype=dtype)[:, None, None]

    log_prob = F.log_softmax(vq_logits, dim=-1)
    target_onehot = F.one_hot(target_vq, K).to(dtype=dtype)

    if mode == "uniform":
        smooth_dist = torch.full_like(target_onehot, 1.0 / K)

    elif mode == "same_aa":
        if aa_target is None:
            raise ValueError('aa_target is required when mode="same_aa"')
        if vq_to_seq_matrix is None:
            raise ValueError('vq_to_seq_matrix is required when mode="same_aa"')

        aa_target = torch.clamp(aa_target.long(), 0, vq_to_seq_matrix.shape[-1] - 1)
        M = vq_to_seq_matrix.to(device=device, dtype=dtype)

        same_aa_mask = M[:, aa_target.reshape(-1)].T
        same_aa_mask = same_aa_mask.reshape(B, L, K)

        denom = same_aa_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        smooth_dist = same_aa_mask / denom

    else:
        raise ValueError(f"Unknown mode: {mode}")

    soft_target = (1.0 - eps) * target_onehot + eps * smooth_dist
    loss_raw = -(soft_target * log_prob).sum(dim=-1)

    if not reduce:
        return loss_raw

    loss = (loss_raw * mask).sum() / (mask.sum() + 1e-8)
    return loss
