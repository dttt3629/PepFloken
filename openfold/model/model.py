# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import torch
import torch.nn as nn

from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    atom14_to_atom37,
)
from openfold.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    TemplateAngleEmbedder,
    TemplatePairEmbedder,
    ExtraMSAEmbedder,
)
from openfold.model.evoformer import EvoformerStack, ExtraMSAStack
from openfold.model.heads import AuxiliaryHeads
import openfold.np.residue_constants as residue_constants
from openfold.model.structure_module import StructureModule
from openfold.model.template import (
    TemplatePairStack,
    TemplatePointwiseAttention,
)
from openfold.utils.loss import (
    compute_plddt,
)
from openfold.utils.tensor_utils import (
    dict_multimap,
    tensor_tree_map,
)


class AlphaFold(nn.Module):


    def __init__(self, config):

        super(AlphaFold, self).__init__()

        self.globals = config.globals
        config = config.model
        template_config = config.template
        extra_msa_config = config.extra_msa


        self.input_embedder = InputEmbedder(
            **config["input_embedder"],
        )
        self.recycling_embedder = RecyclingEmbedder(
            **config["recycling_embedder"],
        )
        self.template_angle_embedder = TemplateAngleEmbedder(
            **template_config["template_angle_embedder"],
        )
        self.template_pair_embedder = TemplatePairEmbedder(
            **template_config["template_pair_embedder"],
        )
        self.template_pair_stack = TemplatePairStack(
            **template_config["template_pair_stack"],
        )
        self.template_pointwise_att = TemplatePointwiseAttention(
            **template_config["template_pointwise_attention"],
        )
        self.extra_msa_embedder = ExtraMSAEmbedder(
            **extra_msa_config["extra_msa_embedder"],
        )
        self.extra_msa_stack = ExtraMSAStack(
            **extra_msa_config["extra_msa_stack"],
        )
        self.evoformer = EvoformerStack(
            **config["evoformer_stack"],
        )
        self.structure_module = StructureModule(
            **config["structure_module"],
        )

        self.aux_heads = AuxiliaryHeads(
            config["heads"],
        )

        self.config = config

    def embed_templates(self, batch, z, pair_mask, templ_dim): 

        template_embeds = []
        n_templ = batch["template_aatype"].shape[templ_dim]
        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx),
                batch,
            )

            single_template_embeds = {}
            if self.config.template.embed_angles:
                template_angle_feat = build_template_angle_feat(
                    single_template_feats,
                )


                a = self.template_angle_embedder(template_angle_feat)

                single_template_embeds["angle"] = a


            t = build_template_pair_feat(
                single_template_feats,
                inf=self.config.template.inf,
                eps=self.config.template.eps,
                **self.config.template.distogram,
            ).to(z.dtype)
            t = self.template_pair_embedder(t)

            single_template_embeds.update({"pair": t})

            template_embeds.append(single_template_embeds)

        template_embeds = dict_multimap(
            partial(torch.cat, dim=templ_dim),
            template_embeds,
        )


        t = self.template_pair_stack(
            template_embeds["pair"], 
            pair_mask.unsqueeze(-3).to(dtype=z.dtype), 
            chunk_size=self.globals.chunk_size,
            _mask_trans=self.config._mask_trans,
        )


        t = self.template_pointwise_att(
            t, 
            z, 
            template_mask=batch["template_mask"].to(dtype=z.dtype),
            chunk_size=self.globals.chunk_size,
        )
        t = t * (torch.sum(batch["template_mask"]) > 0)

        ret = {}
        if self.config.template.embed_angles:
            ret["template_angle_embedding"] = template_embeds["angle"]

        ret.update({"template_pair_embedding": t})

        return ret

    def iteration(self, feats, m_1_prev, z_prev, x_prev, _recycle=True):

        outputs = {}


        dtype = next(self.parameters()).dtype
        for k in feats:
            if(feats[k].dtype == torch.float32):
                feats[k] = feats[k].to(dtype=dtype)


        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device


        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]





        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
        )


        if None in [m_1_prev, z_prev, x_prev]:

            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )


            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.c_z),
                requires_grad=False,
            )


            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

        x_prev = pseudo_beta_fn(
            feats["aatype"], x_prev, None
        ).to(dtype=z.dtype)



        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
        )





        if(not _recycle):
            m_1_prev_emb *= 0
            z_prev_emb *= 0


        m[..., 0, :, :] += m_1_prev_emb


        z += z_prev_emb


        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb


        if self.config.template.enabled:
            template_feats = {
                k: v for k, v in feats.items() if k.startswith("template_")
            }
            template_embeds = self.embed_templates(
                template_feats,
                z,
                pair_mask.to(dtype=z.dtype),
                no_batch_dims,
            )


            z = z + template_embeds["template_pair_embedding"]

            if self.config.template.embed_angles:

                m = torch.cat(
                    [m, template_embeds["template_angle_embedding"]], 
                    dim=-3
                )


                torsion_angles_mask = feats["template_torsion_angles_mask"]
                msa_mask = torch.cat(
                    [feats["msa_mask"], torsion_angles_mask[..., 2]], 
                    dim=-2
                )


        if self.config.extra_msa.enabled:

            a = self.extra_msa_embedder(build_extra_msa_feat(feats))


            z = self.extra_msa_stack(
                a,
                z,
                msa_mask=feats["extra_msa_mask"].to(dtype=a.dtype),
                chunk_size=self.globals.chunk_size,
                pair_mask=pair_mask.to(dtype=z.dtype),
                _mask_trans=self.config._mask_trans,
            )





        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=self.globals.chunk_size,
            _mask_trans=self.config._mask_trans,
        )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s


        outputs["sm"] = self.structure_module(
            s,
            z,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]




        m_1_prev = m[..., 0, :, :]


        z_prev = z


        x_prev = outputs["final_atom_positions"]

        return outputs, m_1_prev, z_prev, x_prev

    def _disable_activation_checkpointing(self):
        self.template_pair_stack.blocks_per_ckpt = None
        self.evoformer.blocks_per_ckpt = None

        for b in self.extra_msa_stack.blocks:
            b.ckpt = False

    def _enable_activation_checkpointing(self):
        self.template_pair_stack.blocks_per_ckpt = (
            self.config.template.template_pair_stack.blocks_per_ckpt
        )
        self.evoformer.blocks_per_ckpt = (
            self.config.evoformer_stack.blocks_per_ckpt
        )

        for b in self.extra_msa_stack.blocks:
            b.ckpt = self.config.extra_msa.extra_msa_stack.ckpt

    def forward(self, batch):


        m_1_prev, z_prev, x_prev = None, None, None


        is_grad_enabled = torch.is_grad_enabled()
        self._disable_activation_checkpointing()


        num_iters = batch["aatype"].shape[-1]
        for cycle_no in range(num_iters):

            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)


            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    self._enable_activation_checkpointing()

                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()


                outputs, m_1_prev, z_prev, x_prev = self.iteration(
                    feats,
                    m_1_prev,
                    z_prev,
                    x_prev,
                    _recycle=(num_iters > 1)
                )


        outputs.update(self.aux_heads(outputs))

        return outputs
