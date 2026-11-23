import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import math
from torch.nn.utils.rnn import pad_sequence
class ResidualLayer(nn.Module):


    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                    stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                    stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):


    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x
class Encoder(nn.Module):


    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.mlp1=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),          
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),          
            nn.GELU(),
        )
        self.mlp2=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),          
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),          
            nn.GELU(),
        )


    def forward(self, x):

        x = self.mlp1(x)+x
        x = self.mlp2(x)+x
        return x
class Decoder(nn.Module):


    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()

        self.mlp1=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),          
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),          
            nn.GELU(),
        )
        self.mlp2=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),          
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),          
            nn.GELU(),
        )
        self.activation = nn.GELU()

    def forward(self, x):

        x = self.mlp1(x)+x
        x = self.mlp2(x)+x
        return x
class AttentionDecoder(nn.Module):

    def __init__(self, latent_dim=64, hidden_dim=128, output_dim=256, num_heads=4, num_layers=2):
        super(AttentionDecoder, self).__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True  
            ) for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=768)

    def forward(self, x,mask):

        x = self.input_proj(x)  

        x = self.pos_encoder(x)  

        for layer in self.attention_layers:
            x = layer(x,src_key_padding_mask=~mask)  

        x = self.output_proj(x)  

        return x
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)

class VectorQuantizer2d(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer2d, self).__init__()

        self.num_embeddings = num_embeddings  
        self.embedding_dim = embedding_dim    
        self.commitment_cost = commitment_cost  

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self._ema_w = nn.Parameter(torch.zeros_like(self.embeddings.weight))
        self.embeddings.weight.data.normal_(0, 0.02)
    def forward(self, inputs):
        r,t = inputs.shape
        flat_input=inputs

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embeddings.weight) 

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encoding_indices
class VectorQuantizer2dema(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost,decay=0.99, epsilon=1e-5):
        super(VectorQuantizer2dema, self).__init__()

        self.num_embeddings = num_embeddings  
        self.embedding_dim = embedding_dim    
        self.commitment_cost = commitment_cost  
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._decay =decay
        self._epsilon = epsilon
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.normal_(0, 0.02)
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim ))
        self._ema_w.data.normal_()
        self.embeddings.weight.requires_grad = False  
        self._ema_w.requires_grad = False  
        self.use_counts = torch.zeros(num_embeddings)
    def forward(self, inputs):
        r,t = inputs.shape
        flat_input=inputs

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embeddings.weight) 
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                    (1 - self._decay) * torch.sum(encodings, 0)

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self.embeddings.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))




        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encoding_indices
class Prework(nn.Module):
    def __init__(self, input_dim=128,out_dim= 256):
        super(Prework, self).__init__()
        self.seqemb =nn.Embedding(31,31)
        self.mlp1 = nn.Linear(39,out_dim)
    def forward(self,resls,chi_angs):

        resls = self.seqemb(resls)
        output = self.mlp1(torch.concat([resls.squeeze(),chi_angs],dim=1))
        return output
class CustomVQVAE(nn.Module):

    def __init__(self, 
                input_dim=256, 
                hidden_dim=512, 
                embedding_dim=512, 
                num_embeddings=512, 
                commitment_cost=2,
                num_attention_heads=4,
                num_attention_layers=2,
                ema= False):
        super(CustomVQVAE, self).__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=embedding_dim
        )
        if ema:
            self.vq_layer = VectorQuantizer2dema(
                num_embeddings=num_embeddings,
                embedding_dim=hidden_dim,
                commitment_cost=commitment_cost
            )
        else:
                self.vq_layer = VectorQuantizer2d(
                num_embeddings=num_embeddings,
                embedding_dim=hidden_dim,
                commitment_cost=commitment_cost
            )

        self.decoder = Decoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=embedding_dim
        )
        self.prework = Prework(input_dim=input_dim,out_dim=hidden_dim)
        self.seqhead= nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 31),

        )
        self.anghead =nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.Tanh()
        )
        self.num_embeddings= num_embeddings
        self.kaiming_init()
        self.orthogonal_init(self.prework.seqemb.weight.data)
    def  orthogonal_init(self,weight):

                nn.init.uniform_(weight, -1, 1)
    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  

            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.01, 0.01)


            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def decode(self,encoding_indices):
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1) 
        quantized = torch.matmul(encodings,self.vq_layer.embeddings.weight)
        x_recon = self.decoder(quantized)
        x_seq = self.seqhead(x_recon)
        x_ang = self.anghead(x_recon)
        return x_recon,x_seq,x_ang
    def forward(self, resls,chi_angs):
        x =self.prework(resls,chi_angs)
        z = self.encoder(x)  

        quantized, vq_loss, perplexity, encoding_indices = self.vq_layer(z)

        x_recon = self.decoder(quantized)
        x_seq = self.seqhead(x_recon)
        x_ang = self.anghead(x_recon)
        return x_recon,x_seq,x_ang, vq_loss, perplexity, encoding_indices
def pads(ten,vlen):
    maxlen= max(vlen)
    z=0
    tsls = []
    for v in vlen:
        tsls.append(ten[z:z+v])
        z+=v
    padded_tensors = pad_sequence(
        tsls,
        batch_first=True,    
        padding_value=0      
    )

    mask = (padded_tensors != 0).any(dim=-1) 
    return padded_tensors,mask
class Con_CustomVQVAE(nn.Module):

    def __init__(self, 
                input_dim=256, 
                hidden_dim=128, 
                embedding_dim=64, 
                num_embeddings=512, 
                commitment_cost=2,
                num_attention_heads=4,
                num_attention_layers=2,
                ema = False):
        super(Con_CustomVQVAE, self).__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=embedding_dim
        )

        if ema:
            self.vq_layer = VectorQuantizer2dema(
                num_embeddings=num_embeddings,
                embedding_dim=hidden_dim,
                commitment_cost=commitment_cost
            )
        else:
                self.vq_layer = VectorQuantizer2d(
                num_embeddings=num_embeddings,
                embedding_dim=hidden_dim,
                commitment_cost=commitment_cost
            )

        self.decoder = AttentionDecoder(
            latent_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_attention_heads,
            num_layers=num_attention_layers
        )
        self.prework = Prework(input_dim=input_dim,out_dim=hidden_dim)
        self.seqhead= nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 31),
        )
        self.anghead =nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.Tanh()
        )
        self.tranform = nn.Linear(6,hidden_dim)
        self.kaiming_init()
        self.orthogonal_init(self.prework.seqemb.weight.data)
    def  orthogonal_init(self,weight):

                nn.init.uniform_(weight, -1, 1)
    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  

            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.01, 0.01)


            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def get_ind(self,resls,chi_angs):
        x =self.prework(resls,chi_angs)
        z = self.encoder(x)  

        quantized, vq_loss, perplexity, encoding_indices = self.vq_layer(z)
        return encoding_indices
    def decode(self,encoding_indices,ang):
        encodings = torch.zeros(encoding_indices.shape[0], self.vq_layer.num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.vq_layer.embeddings.weight)
        x_recon = self.decoder(quantized+self.tranform(ang),mask)
        x_seq = self.seqhead(x_recon)
        x_ang = self.anghead(x_recon)
        return x_recon,x_seq,x_ang
    def forward(self,  resls,chi_angs,ang,vlen):
        x =self.prework(resls,chi_angs)
        z = self.encoder(x)  

        quantized, vq_loss, perplexity, encoding_indices = self.vq_layer(z)

        quantized,mask=pads(quantized,vlen)
        ang,_ =pads(ang.reshape(-1,6),vlen)

        x_recon = self.decoder(quantized+self.tranform(ang),mask)
        x_seq = self.seqhead(x_recon)
        x_ang = self.anghead(x_recon)
        return x_recon,x_seq,x_ang, vq_loss, perplexity, encoding_indices,mask
class Con_CustomVQVAE2(nn.Module):

    def __init__(self, 
                input_dim=256, 
                hidden_dim=128, 
                embedding_dim=64, 
                num_embeddings=512, 
                commitment_cost=0.25,
                num_attention_heads=4,
                num_attention_layers=2,
                ema = False):
        super(Con_CustomVQVAE2, self).__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=embedding_dim
        )

        if ema:
            self.vq_layer = VectorQuantizer2dema(
                num_embeddings=num_embeddings,
                embedding_dim=hidden_dim,
                commitment_cost=commitment_cost
            )
        else:
                self.vq_layer = VectorQuantizer2d(
                num_embeddings=num_embeddings,
                embedding_dim=hidden_dim,
                commitment_cost=commitment_cost
            )

        self.decoder = AttentionDecoder(
            latent_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_attention_heads,
            num_layers=num_attention_layers
        )
        self.prework = Prework(input_dim=input_dim,out_dim=hidden_dim)
        self.seqhead= nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 31),
        )
        self.poshead =nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 45),
        )
        self.tranform = nn.Linear(6,hidden_dim)
        self.kaiming_init()
        self.orthogonal_init(self.prework.seqemb.weight.data)
    def  orthogonal_init(self,weight):

                nn.init.uniform_(weight, -1, 1)
    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  

            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.01, 0.01)


            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def get_ind(self,resls,chi_angs):
        x =self.prework(resls,chi_angs)
        z = self.encoder(x)  

        quantized, vq_loss, perplexity, encoding_indices = self.vq_layer(z)
        return encoding_indices
    def decode(self,encoding_indices,ang,mask=None):
        encodings = torch.zeros(encoding_indices.shape[0], self.vq_layer.num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.vq_layer.embeddings.weight)
        x_recon = self.decoder(quantized+self.tranform(ang),mask)
        x_seq = self.seqhead(x_recon)
        x_pos= self.poshead(x_recon)
        return x_recon,x_seq,x_pos
    def forward(self,  resls,chi_angs,ang,vlen):
        x =self.prework(resls,chi_angs)
        z = self.encoder(x)  

        quantized, vq_loss, perplexity, encoding_indices = self.vq_layer(z)

        quantized,mask=pads(quantized,vlen)
        ang,_ =pads(ang.reshape(-1,6),vlen)

        x_recon = self.decoder(quantized+self.tranform(ang),mask)
        x_seq = self.seqhead(x_recon)
        x_pos = 8.0 * torch.tanh(self.poshead(x_recon))
        return x_recon,x_seq,x_pos, vq_loss, perplexity, encoding_indices,mask
