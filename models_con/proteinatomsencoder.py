from functools import partial

import torch
from torch import nn
from torch.nn import Module
from torch.nn.functional import one_hot
import torch.nn.functional as F
from models_con.utils import get_time_embedding
def get_indexing_matrix(K, W, H, device):
    assert W % 2 == 0
    assert H % (W // 2) == 0

    h = H // (W // 2)
    assert h % 2 == 0

    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(
        min=0, max=h + 1
    )
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()


def single_to_keys(single, indexing_matrix, W, H):

    B, N, D = single.shape
    K = N // W
    single = single.view(B, 2 * K, W // 2, D)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )
import torch
import torch.nn.functional as F

def pack_atoms_like_boltz(x_global_t, atom_mask, elementindex=None, t=None):

    B, N, A, _ = x_global_t.shape
    device = x_global_t.device

    packed_pos = []
    packed_elem = []
    packed_res_idx = []
    packed_t = []
    n_atoms_list = []

    for b in range(B):
        valid = atom_mask[b].bool()                 
        n_atoms = int(valid.sum().item())
        n_atoms_list.append(n_atoms)

        pos_b = x_global_t[b][valid]
        packed_pos.append(pos_b)

        res_idx = (
            torch.arange(N, device=device)[:, None]
            .expand(N, A)[valid]
        )  
        packed_res_idx.append(res_idx)

        if elementindex is not None:
            elem_b = elementindex[b][valid]         
            packed_elem.append(elem_b)

        if t is not None:
            t_b = t[b:b+1].expand(n_atoms, t.shape[-1])   
            packed_t.append(t_b)

    N_atom_max = max(n_atoms_list)

    ref_pos = x_global_t.new_zeros(B, N_atom_max, 3)
    atom_pad_mask = torch.zeros(B, N_atom_max, device=device, dtype=torch.bool)
    atom_to_token = torch.zeros(B, N_atom_max, N, device=device, dtype=x_global_t.dtype)
    atom_res_idx = torch.full((B, N_atom_max), -1, device=device, dtype=torch.long)

    ref_element = None
    if elementindex is not None:
        ref_element = torch.zeros(B, N_atom_max, device=device, dtype=elementindex.dtype)

    t_atom = None
    if t is not None:
        t_atom = torch.zeros(B, N_atom_max, t.shape[-1], device=device, dtype=t.dtype)

    for b in range(B):
        n_atoms = n_atoms_list[b]

        ref_pos[b, :n_atoms] = packed_pos[b]
        atom_pad_mask[b, :n_atoms] = True
        atom_res_idx[b, :n_atoms] = packed_res_idx[b]

        atom_to_token[b, :n_atoms] = F.one_hot(
            packed_res_idx[b], num_classes=N
        ).to(x_global_t.dtype)

        if ref_element is not None:
            ref_element[b, :n_atoms] = packed_elem[b]

        if t_atom is not None:
            t_atom[b, :n_atoms] = packed_t[b]

    return {
        "ref_pos": ref_pos,
        "atom_pad_mask": atom_pad_mask,
        "atom_to_token": atom_to_token,
        "atom_res_idx": atom_res_idx,
        "ref_element": ref_element,
        "t_atom": t_atom,
        "n_atoms": torch.tensor(n_atoms_list, device=device),
    }


class AdaLN(nn.Module):


    def __init__(self, dim, dim_single_cond):

        super().__init__()
        self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(dim_single_cond, bias=False)
        self.s_scale = nn.Linear(dim_single_cond, dim)
        self.s_bias = nn.Linear(dim_single_cond, dim,bias=False)

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = torch.sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a

class PairBiasedAttention(nn.Module):
    def __init__(
        self,
        node_dim: int,
        pair_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        inf: float = 1e6,
        initial_norm: bool = True,
    ):
        super().__init__()
        assert node_dim % n_heads == 0

        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.inf = inf
        self.initial_norm = initial_norm

        if self.initial_norm:
            self.node_norm = nn.LayerNorm(node_dim)

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.v_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.o_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.proj_g = nn.Linear(node_dim, node_dim, bias=False)
        self.pair_bias_proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, n_heads, bias=False),
        )

        self.dropout = nn.Dropout(dropout)
        self.adaln = AdaLN(node_dim, node_dim)

    def forward(
        self,
        a,
        s,
        z,
        mask: torch.Tensor = None,
        to_keys=None,

        ) -> torch.Tensor:

        Bx = a.shape[0]
        x = self.adaln(a, s)
        x = self.node_norm (x)
        pair_bias = self.pair_bias_proj(z).permute(0, 3, 1, 2)

        if to_keys is not None:
            k_in = to_keys(x)  
            if mask is not None:
                mask_k = to_keys(mask.unsqueeze(-1).float()).squeeze(-1).bool()  
            else:
                mask_k = None
        else:
            k_in = x
            mask_k = mask

        q = self.q_proj(x).view(Bx, -1, self.n_heads, self.head_dim).transpose(1, 2)

        k = self.k_proj(k_in).view(Bx, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(k_in).view(Bx, -1, self.n_heads, self.head_dim).transpose(1, 2)


        g = self.proj_g(x).sigmoid()

        with torch.autocast("cuda", enabled=False):
            q = q.float()
            k = k.float()
            v = v.float()
            pair_bias = pair_bias.float()

            logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            logits = logits + pair_bias

            if mask_k is not None:
                logits = logits + (1.0 - mask_k[:, None, None].float()) * (-self.inf)

            attn = F.softmax(logits, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)  
            out = out.to(dtype=x.dtype)

        out = out.transpose(1, 2).contiguous().view(Bx, -1, self.node_dim)
        out = self.o_proj(g*out)
        return out

class AtomTransformer(Module):


    def __init__(
        self,
        atom_s=128,
        atom_z=64,
        attn_window_queries=None,
        attn_window_keys=None,
        depth=3,
        atom_encoder_heads=4,
        dropout=0.0,
    ):

        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys
        self.layers = nn.ModuleList()
        for _ in range(depth):
                            self.layers.append(
                        PairBiasedAttention(
                            node_dim=atom_s,
                            pair_dim=atom_z,
                           n_heads = atom_encoder_heads,
                           dropout=dropout,
                        ),

                    )
    def forward(
        self,
        q,
        c,
        p,
        to_keys=None,
        mask=None,
    ):
        W = self.attn_window_queries
        H = self.attn_window_keys

        if W is not None:
            B, N, D = q.shape
            NW = N // W

            q = q.view((B * NW, W, -1))
            c = c.view((B * NW, W, -1))
            if mask is not None:
                mask = mask.view(B * NW, W)
            p = p.view((p.shape[0] * NW, W, H, -1))

            to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)
        else:
            to_keys_new = None

        for layer in self.layers:
            q = q+layer(
                a=q,
                s=c,
                z=p,
                mask=mask.float() if mask is not None else None,
                to_keys=to_keys_new,
            )

        if W is not None:
            q = q.view((B, NW * W, D))

        return q
class ProteinAtomAttentionEncoder(nn.Module):


    def __init__(
        self,
        atom_s=128,
        atom_z=64,
        token_s=128,
        token_z = 64,
        atoms_per_window_queries=32,
        atoms_per_window_keys=64,
        num_element_types=6,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        structure_prediction= False,
        activation_checkpointing=False,
    ):
        super().__init__()

        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.structure_prediction = structure_prediction

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys

        self.element_embed = nn.Embedding(num_element_types, atom_s)
        self.embed_atom_pos = nn.Linear(3, atom_s, bias=False)
        self.atom_pos_proj = nn.Linear(atom_s, atom_s, bias=False)
        self.atom_elem_proj = nn.Linear(atom_s, atom_s, bias=False)
        self.atom_t_proj = nn.Linear(atom_s, atom_s, bias=False)
        self.embed_atompair_ref_pos = nn.Linear(3, atom_z, bias=False)
        self.embed_atompair_ref_dist = nn.Linear(1, atom_z, bias=False)
        self.embed_atompair_mask = nn.Linear(1, atom_z, bias=False)


        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_s, atom_z, bias=False),
        ) 

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_s, atom_z, bias=False),
        )


        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(atom_z, atom_z, bias=False),      
            nn.ReLU(),
            nn.Linear(atom_z, atom_z, bias=False),
            nn.ReLU(),
            nn.Linear(atom_z, atom_z, bias=False),
        )


        self.atom_encoder = AtomTransformer(
        atom_s=atom_s,
        atom_z=atom_z,
        attn_window_queries=atoms_per_window_queries,
        attn_window_keys=atoms_per_window_keys,
        depth=atom_encoder_depth,
        atom_encoder_heads=atom_encoder_heads,
        dropout=0.0,
        )

        self.atom_to_token_trans = nn.Sequential(
            nn.Linear(atom_s, 2 * token_s if structure_prediction else token_s, bias=False),
            nn.ReLU(),
        )



    @staticmethod
    def pad_atoms_to_window_size(x, multiple, value=0):

        B, L = x.shape[:2]
        pad_len = (multiple - (L % multiple)) % multiple
        if pad_len == 0:
            return x, pad_len

        pad_shape = list(x.shape)
        pad_shape[1] = pad_len
        pad_tensor = torch.full(
            pad_shape, value, dtype=x.dtype, device=x.device
        )
        x = torch.cat([x, pad_tensor], dim=1)
        return x, pad_len

    def embed_t(self, timesteps, n_tokens):
        t_emb = get_time_embedding(
            timesteps[:, 0],   
            self.atom_s,
            max_positions=2056
        )  

        t_emb = t_emb[:, None, :].expand(-1, n_tokens, -1)  
        return t_emb
    def forward(
        self,
        x_global_t,
        atom_mask,
        t,
        elementindex,
        s_trunk=None,   
        z=None,         
        r=None,         
        model_cache=None,
        return_atom_state = False,

    ):
        packed = pack_atoms_like_boltz(x_global_t, atom_mask, elementindex, t)

        atom_ref_pos = packed["ref_pos"]              
        atom_mask_flat = packed["atom_pad_mask"]      
        atom_to_token = packed["atom_to_token"]       
        elementindex_flat = packed["ref_element"]     
        atom_res_uid = packed["atom_res_idx"]         

        B, N, A, _ = x_global_t.shape
        device = x_global_t.device




        W = self.atoms_per_window_queries
        H = self.atoms_per_window_keys

        atom_ref_pos, pad_len = self.pad_atoms_to_window_size(atom_ref_pos, W, value=0)
        atom_mask_flat, _ = self.pad_atoms_to_window_size(
            atom_mask_flat, W, value=False
        )
        elementindex_flat, _ = self.pad_atoms_to_window_size(
            elementindex_flat, W, value=0
        )
        atom_res_uid, _ = self.pad_atoms_to_window_size(atom_res_uid, W, value=-1)
        atom_to_token, _ = self.pad_atoms_to_window_size(atom_to_token, W, value=0)

        B, N_atom, _ = atom_ref_pos.shape
        K = N_atom // W



        keys_indexing_matrix = get_indexing_matrix(K, W, H, atom_ref_pos.device)
        to_keys = partial(
            single_to_keys,
            indexing_matrix=keys_indexing_matrix,
            W=W,
            H=H,
        )
        atom_pos_feat = self.embed_atom_pos(atom_ref_pos)              
        atom_elem_feat = self.element_embed(elementindex_flat)         
        t_feat = self.embed_t(t, atom_ref_pos.shape[1])
        c = (
            self.atom_pos_proj(atom_pos_feat)
            + self.atom_elem_proj(atom_elem_feat)
            + self.atom_t_proj(t_feat)
        )
        atom_ref_pos_queries = atom_ref_pos.view(B, K, W, 1, 3)
        atom_ref_pos_keys = to_keys(atom_ref_pos).view(B, K, 1, H, 3)

        d = atom_ref_pos_keys - atom_ref_pos_queries                  
        d_norm = torch.sum(d * d, dim=-1, keepdim=True)
        d_norm = 1.0 / (1.0 + d_norm)

        atom_mask_queries = atom_mask_flat.view(B, K, W, 1)
        atom_mask_keys = to_keys(atom_mask_flat.unsqueeze(-1).float()).view(B, K, 1, H).bool()

        atom_uid_queries = atom_res_uid.view(B, K, W, 1)
        atom_uid_keys = to_keys(atom_res_uid.unsqueeze(-1).float()).view(B, K, 1, H).long()

        v = (
            (
                atom_mask_queries
                & atom_mask_keys
                & (atom_uid_queries == atom_uid_keys)
            )
            .float()
            .unsqueeze(-1)
        )


        p = self.embed_atompair_ref_pos(d) * v
        p = p + self.embed_atompair_ref_dist(d_norm) * v
        p = p + self.embed_atompair_mask(v) * v

        q = c
        p = p + self.c_to_p_trans_q(c.view(B, K, W, 1, c.shape[-1]))
        p = p + self.c_to_p_trans_k(to_keys(c).view(B, K, 1, H, c.shape[-1]))
        p = p + self.p_mlp(p)
        q = self.atom_encoder(
            q=q,
            mask=atom_mask_flat,
            c=c,
            p=p,
            to_keys=to_keys,

        )
        atom_state = None
        if return_atom_state:
            atom_state = {
                "q": q,
                "c": c,
                "p": p,
                "to_keys": to_keys,
                "atom_to_token": atom_to_token,
                "atom_pad_mask": atom_mask_flat,
                "atom_res_idx": atom_res_uid,
                "pad_len": pad_len,
            }

        q_to_a = self.atom_to_token_trans(q)
        atom_to_token_masked = atom_to_token * atom_mask_flat[:, :, None].float()

        if pad_len > 0:
            atom_to_token_masked = atom_to_token_masked[:, :-pad_len]
            q_to_a = q_to_a[:, :-pad_len]

        atom_to_token_mean = atom_to_token_masked / (
            atom_to_token_masked.sum(dim=1, keepdim=True) + 1e-6
        )
        a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)
        if pad_len > 0:
            q = q[:, :-pad_len]
            c = c[:, :-pad_len]

        if return_atom_state:
            return a, atom_state

        return a, q, c, p, to_keys
class AtomAttentionDecoder(nn.Module):
    def __init__(
        self,
        atom_s,
        token_s,
        token_pair_s,
        atom_z,
        attn_window_queries=32,
        attn_window_keys=64,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
    ):
        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys

        self.token_to_atom_q = nn.Linear(token_s, atom_s, bias=False)
        self.token_to_atom_c = nn.Linear(token_s, atom_s, bias=False)
        self.token_pair_to_atom_pair = nn.Linear(token_pair_s, atom_z, bias=False)

        self.atom_decoder = AtomTransformer(
            atom_s=atom_s,
            atom_z=atom_z,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            depth=atom_decoder_depth,
            atom_encoder_heads=atom_decoder_heads,
        )
        self.atom_feat_to_pos_update = nn.Sequential(
            nn.LayerNorm(atom_s),
            nn.Linear(atom_s, 3, bias=False),
        )

    def residue_pair_to_atom_pair(self, pair, atom_state):
        atom_res_idx = atom_state["atom_res_idx"]
        atom_mask = atom_state["atom_pad_mask"]
        to_keys = atom_state["to_keys"]

        B, M_pad = atom_res_idx.shape
        W = self.attn_window_queries
        H = self.attn_window_keys

        assert M_pad % W == 0
        K = M_pad // W

        query_res = atom_res_idx.view(B, K, W)
        key_res = to_keys(atom_res_idx.unsqueeze(-1).float()).view(B, K, H).long()

        query_mask = atom_mask.view(B, K, W)
        key_mask = to_keys(atom_mask.unsqueeze(-1).float()).view(B, K, H).bool()

        valid = (
            query_mask[..., None]
            & key_mask[:, :, None, :]
            & (query_res[..., None] >= 0)
            & (key_res[:, :, None, :] >= 0)
        )

        query_res = query_res.clamp(min=0, max=pair.shape[1] - 1)
        key_res = key_res.clamp(min=0, max=pair.shape[2] - 1)

        batch_idx = torch.arange(B, device=pair.device).view(B, 1, 1, 1)
        atom_pair = pair[
            batch_idx,
            query_res[..., None],
            key_res[:, :, None, :],
        ]

        atom_pair = self.token_pair_to_atom_pair(atom_pair)
        return atom_pair * valid[..., None].to(atom_pair.dtype)

    def unpack_atom_update(self, flat_update, atom_mask, out_shape):
        B, N, A = out_shape
        update = flat_update.new_zeros(B, N, A, 3)
        for b in range(B):
            valid = atom_mask[b].bool()
            n_valid = int(valid.sum().item())
            update[b][valid] = flat_update[b, :n_valid]
        return update

    def forward(self, token_feat, pair, atom_state, atom_mask):
        atom_to_token = atom_state["atom_to_token"].float()

        atom_token_q = torch.bmm(atom_to_token, self.token_to_atom_q(token_feat.float()))
        atom_token_c = torch.bmm(atom_to_token, self.token_to_atom_c(token_feat.float()))
        q = atom_state["q"] + atom_token_q.to(atom_state["q"].dtype)
        c = atom_state["c"] + atom_token_c.to(atom_state["c"].dtype)
        p = atom_state["p"] + self.residue_pair_to_atom_pair(pair, atom_state).to(
            atom_state["p"].dtype
        )

        q = self.atom_decoder(
            q=q,
            c=c,
            p=p,
            mask=atom_state["atom_pad_mask"],
            to_keys=atom_state["to_keys"],
        )
        flat_update = self.atom_feat_to_pos_update(q)
        return self.unpack_atom_update(
            flat_update,
            atom_mask,
            (atom_mask.shape[0], atom_mask.shape[1], atom_mask.shape[2]),
        )
