import math
import torch
from torch import nn
import torch.nn.functional as F

from models_con.utils import get_time_embedding
from pepfloken.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms

from con_code.common.residue import atom_element,restype_name_to_atom14_names,atom_types
from models_con.proteinatomsencoder import ProteinAtomAttentionEncoder



class AAtoElementIndex(nn.Module):


    def __init__(
        self,out_dtype=torch.long,
    ):
        super().__init__()
        atoms_set = list(set(atom_element.values()))
        atoms_dic = {v:k for k,v in enumerate(atoms_set)}
        element_dic = {k:atoms_dic[atom_element[k]] for k in atom_element.keys()}
        RESTYPE_ORDER = {aa.name: aa.value for aa in AA}
        RESTYPE_ORDER['PAD']=21
        self.restype_order = RESTYPE_ORDER
        self.pad_value = len(atoms_dic)+1
        num_restypes = max(self.restype_order.values()) + 1
        atom14_size = 15

        table = torch.full(
            (num_restypes, atom14_size),
            fill_value=self.pad_value,
            dtype=out_dtype,
        )

        for resname, res_idx in RESTYPE_ORDER.items():
            if resname in restype_name_to_atom14_names:
                atom_names = restype_name_to_atom14_names.get(
                    resname,
                    restype_name_to_atom14_names.get(resname, [""] * atom14_size),
                )
                for atom_slot, atom_name in enumerate(atom_names):
                    if atom_name == "":
                        continue

                    if atom_name not in element_dic:
                        raise KeyError(f"Atom name {atom_name} not found in element_dic")

                    table[res_idx, atom_slot] = element_dic[atom_name]


        self.register_buffer("aa_to_element_table", table, persistent=False)

    def forward(self, aa_index: torch.Tensor):

        if aa_index.dtype != torch.long:
            aa_index = aa_index.long()

        element_index = self.aa_to_element_table[aa_index]   

        return element_index

class PairBiasedAttention(nn.Module):
    def __init__(self, node_dim, pair_dim, n_heads=8, dropout=0.05):
        super().__init__()
        assert node_dim % n_heads == 0
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.o_proj = nn.Linear(node_dim, node_dim)

        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_bias_proj = nn.Linear(pair_dim, n_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair, mask=None):

        B, N, C = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  

        pair_bias = self.pair_bias_proj(self.pair_norm(pair))        
        pair_bias = pair_bias.permute(0, 3, 1, 2).contiguous()      
        logits = logits + pair_bias

        if mask is not None:
            pair_mask = mask[:, None, :, None] & mask[:, None, None, :]   
            logits = logits.masked_fill(~pair_mask, -1e9)

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                 
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.o_proj(out)
        return out


class PairUpdate(nn.Module):
    def __init__(self, node_dim, pair_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or pair_dim * 2

        self.node_norm = nn.LayerNorm(node_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)

        self.left_proj = nn.Linear(node_dim, pair_dim)
        self.right_proj = nn.Linear(node_dim, pair_dim)

        self.mlp_left = nn.Linear(pair_dim, hidden_dim, bias=False)
        self.mlp_right = nn.Linear(pair_dim, hidden_dim, bias=False)
        self.mlp_pair = nn.Linear(pair_dim, hidden_dim, bias=True)

        self.mlp_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pair_dim)
        )

    def forward(self, x, pair, mask=None):

        B, N, _ = x.shape
        x_n = self.node_norm(x)
        pair_n = self.pair_norm(pair)

        left = self.left_proj(x_n)[:, :, None, :]
        right = self.right_proj(x_n)[:, None, :, :]

        h = (
            self.mlp_left(left)
            + self.mlp_right(right)
            + self.mlp_pair(pair_n)
        )
        delta = self.mlp_out(h)

        if mask is not None:
            pair_mask = (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1)  
            delta = delta * pair_mask

        return delta


class Transition(nn.Module):
    def __init__(self, node_dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim, node_dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * mult, node_dim)
        )

    def forward(self, x):
        return self.net(x)


class PairBiasedBlock(nn.Module):
    def __init__(self, node_dim, pair_dim, n_heads=8, dropout=0.0,last_layer=False):
        super().__init__()
        self.node_norm1 = nn.LayerNorm(node_dim)
        self.node_norm2 = nn.LayerNorm(node_dim)

        self.attn = PairBiasedAttention(
            node_dim=node_dim,
            pair_dim=pair_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        self.transition = Transition(
                node_dim=node_dim,
                mult=4,
                dropout=dropout
            )
        self.last_layer = last_layer
        if not last_layer:
            self.pair_update = PairUpdate(
                node_dim=node_dim,
                pair_dim=pair_dim,
                hidden_dim=pair_dim * 2,
                dropout=dropout
            )

            self.pair_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, pair, mask=None):
        x = x + self.attn(self.node_norm1(x), pair, mask=mask)

        if not self.last_layer:
            pair = pair + torch.tanh(self.pair_gate) *self.pair_update(x, pair, mask=mask)

        x = x + self.transition(self.node_norm2(x))

        return x, pair


class SidechainDecoderV2(nn.Module):
    def __init__(
        self,
        node_dim=128,
        edge_dim=64,
        hidden_dim=256,
        n_layers=4,
        n_heads=8,
        max_num_atoms=15,
        max_vq_types=1024,
        max_aa_atoms=22,

    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_num_atoms = max_num_atoms
        self.AAtoElementIndex = AAtoElementIndex()
        self.vq_embed = nn.Embedding(max_vq_types, hidden_dim)
        self.pos_proj = nn.Linear(max_num_atoms * 3, hidden_dim)
        self.aa_embed = nn.Embedding(max_aa_atoms, hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim*2 + hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            PairBiasedBlock(
                node_dim=hidden_dim,
                pair_dim=edge_dim,
                n_heads=n_heads,
                dropout=0.05
            )
            for _ in range(n_layers-1)
        ])
        self.blocks.append(PairBiasedBlock(
            node_dim=hidden_dim,
            pair_dim=edge_dim,
            n_heads=n_heads,
            dropout=0.05,
            last_layer=True
        ))
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim*2, 37 * 3)
        )
        self.atom_encoder = ProteinAtomAttentionEncoder(atom_s = node_dim,
                                            atom_z=edge_dim,
                                            token_s=node_dim,
                                            token_z = edge_dim,)

        self.pair_ai_proj   = nn.Linear(node_dim, hidden_dim, bias=False)
        self.pair_aj_proj   = nn.Linear(node_dim, hidden_dim, bias=False)
        self.pair_sub_proj  = nn.Linear(node_dim, hidden_dim, bias=False)
        self.pair_edge_proj = nn.Linear(edge_dim, hidden_dim, bias=True)

        self.pair_out_proj = nn.Linear(hidden_dim, edge_dim)
        self.pair_act = nn.SiLU()
        self.pair_init_gate = nn.Parameter(torch.zeros(1))
        self.num_rbf = 8
        self.rbf_start = 0.0
        self.rbf_end = 20.0
        RESTYPE_ORDER = {aa.name: aa.value for aa in AA}
        RESTYPE_ORDER['PAD']=21
        atoms37_to_atoms14_matrix = torch.zeros(22,37,15)
        for k,v in RESTYPE_ORDER.items():
            if k !='UNK' and k !='PAD':
                if k in restype_name_to_atom14_names:
                    for i,n in enumerate(restype_name_to_atom14_names[k]):
                        if n == "":
                            continue
                        atoms37_to_atoms14_matrix[v,atom_types.index(n),i] = 1
                    atoms37_to_atoms14_matrix[v,-1,-1] = 1
        self.register_buffer(
            "atoms37_to_atoms14_matrix",
            atoms37_to_atoms14_matrix,
            persistent=False,
        )
        self.register_buffer(
            "rbf_centers",
            torch.linspace(self.rbf_start, self.rbf_end, self.num_rbf),
            persistent=False,
        )

        self.pair_rbf_mlp = nn.Sequential(
            nn.Linear(self.num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        self.pair_rbf_gate = nn.Parameter(torch.zeros(1))

    def compute_residue_centers(self, x_global_t, atom_mask):
        mask = atom_mask[..., None].float()  
        centers = (x_global_t * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1e-6)
        return centers  

    def rbf(self, dist):
        diff = dist.unsqueeze(-1) - self.rbf_centers.view(1, 1, 1, -1)
        gamma = (self.num_rbf / (self.rbf_end - self.rbf_start + 1e-8)) ** 2
        return torch.exp(-gamma * diff.pow(2))  
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.hidden_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def get_pair_feat(self,a,N,edge_embed):
        ai = a[:, :, None, :].expand(-1, -1, N, -1)   
        aj = a[:, None, :, :].expand(-1, N, -1, -1)   

        h = (
            self.pair_ai_proj(ai)
            + self.pair_aj_proj(aj)
            + self.pair_sub_proj(ai - aj)
            + self.pair_edge_proj(edge_embed)
        )   

        h = self.pair_act(h)
        pair_feat = self.pair_out_proj(h)   
        return pair_feat
    def get_rbf_feat(self, x_global_t, atom_mask):
        centers = self.compute_residue_centers(x_global_t, atom_mask)   
        ci = centers[:, :, None, :]   
        cj = centers[:, None, :, :]   
        d = cj - ci                   
        dist = torch.sqrt((d * d).sum(dim=-1) + 1e-8)   
        rbf_feat = self.rbf(dist) 
        rbf_feat  = self.pair_rbf_mlp(rbf_feat)  
        return rbf_feat

    def forward(self, x_global_t, timesteps, vq,aa_index, node_embed, edge_embed, res_mask, gen_mask,atom_mask):
        B, N, _, _ = x_global_t.shape
        element_index = self.AAtoElementIndex(aa_index)

        a, _, _, _, _ = self.atom_encoder(x_global_t, atom_mask,timesteps, element_index)
        z_q = torch.cat([
            node_embed,
            a,
            self.pos_proj(x_global_t.reshape(B, N, -1)),
            self.vq_embed(vq),
            self.aa_embed(aa_index),
            self.embed_t(timesteps, res_mask),
        ], dim=-1)
        x = self.input_proj(z_q)
        pair_feat = self.get_pair_feat(a,N,edge_embed)  
        rbf_feat  = self.get_rbf_feat(x_global_t, atom_mask)  

        pair = (
            edge_embed
            + torch.tanh(self.pair_init_gate) * pair_feat
            + torch.tanh(self.pair_rbf_gate) * rbf_feat
        )

        for i, block in enumerate(self.blocks):
                x, pair = block(x, pair, mask=res_mask.bool())
        pred_atoms = self.output_mlp(self.out_norm(x)).view(B, N, 37, 3)
        mapping  = self.atoms37_to_atoms14_matrix[aa_index]
        pred_atoms = torch.einsum(
            "bnac,bnas->bnsc",
            pred_atoms,
            mapping.to(pred_atoms.dtype),
        )
        if res_mask is not None:
            pred_atoms = pred_atoms * res_mask[..., None, None].float()

        return pred_atoms