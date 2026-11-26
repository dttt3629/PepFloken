from models_con.utils import get_index_embedding, get_time_embedding
import torch
from torch import nn
from torch.nn import functional as F
class CrossAttentionWithDist(nn.Module):

    def __init__(self, embed_dim, bias_dim=64, n_heads=8, dropout=0.0):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.edge_proj = nn.Linear(bias_dim, n_heads)
        self.edge_norm = nn.LayerNorm(bias_dim)
        self.edge_scale = nn.Parameter(torch.tensor(0.2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, edge_emb, mask=None):

        B, Lq, H = query.shape
        _, Lk, _ = key_value.shape

        Q = self.q_proj(query)      
        K = self.k_proj(key_value)  
        V = self.v_proj(key_value)  

        Q = Q.view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)  
        K = K.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)  
        V = V.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)  

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale       


        edge_bias = self.edge_proj(self.edge_norm(edge_emb))                              
        edge_bias = edge_bias.permute(0, 3, 1, 2).contiguous()           
        scores = scores + self.edge_scale * edge_bias

        if mask is not None:
            if mask.dim() == 2:
                mask_exp = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask_exp = mask[:, None, :, :]
            else:
                raise ValueError("mask must have shape [B, Lk] or [B, Lq, Lk]")

            scores = scores.masked_fill(~mask_exp.bool(), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)                          
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)                               

        out = out.transpose(1, 2).contiguous().view(B, Lq, H)            
        out = self.out_proj(out)                                          

        return out, attn_weights
class SidechainDecoder(nn.Module):

    def __init__(self, node_dim=128, edge_dim=64, token_dim=256, ctx_dim=256,
                 hidden_dim=512, n_layers=4, n_heads=8, max_num_atoms=15,
                 rot_noise_std=0.02, trans_noise_std=0.01, max_vq_types=1024,max_aa_types=22):
        super().__init__()
        self.token_dim = token_dim

        self.ctx_dim = ctx_dim
        self.hidden_dim = hidden_dim
        self.max_num_atoms = max_num_atoms
        self.rot_noise_std = rot_noise_std
        self.trans_noise_std = trans_noise_std
        self.vq_embed = nn.Embedding(max_vq_types, hidden_dim)
        self.pos_proj = nn.Linear(45, hidden_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim+hidden_dim*3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True,
            dropout=0.05,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cross_attn = CrossAttentionWithDist(embed_dim=hidden_dim,bias_dim=edge_dim, n_heads=n_heads)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_num_atoms*3)
        )
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.hidden_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, x_local_t, timesteps, vq,node_embed, edge_embed,res_mask,gen_mask):
        B, N, _, _ = x_local_t.shape
        z_q = torch.cat([node_embed,self.pos_proj(x_local_t.reshape(B, N, 45)), self.vq_embed(vq),self.embed_t(timesteps,res_mask)], dim=-1)
        z_q = self.input_proj(z_q)
        x, attn_weights = self.cross_attn(self.attn_norm(z_q), self.attn_norm(z_q),edge_embed, mask=res_mask)
        x=x+z_q
        padding_mask = ~res_mask.bool() 
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        pred_atoms = self.output_mlp(x).view(B, N, self.max_num_atoms, 3)


        if res_mask is not None:
            pred_atoms = pred_atoms * res_mask[..., None, None].float()

        return pred_atoms
