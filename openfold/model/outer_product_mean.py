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
from typing import Optional

import torch
import torch.nn as nn

from openfold.model.primitives import Linear
from openfold.utils.tensor_utils import chunk_layer


class OuterProductMean(nn.Module):


    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):

        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init="final")

    def _opm(self, a, b):

        outer = torch.einsum("...bac,...dae->...bdce", a, b)


        outer = outer.reshape(outer.shape[:-2] + (-1,))


        outer = self.linear_out(outer)

        return outer

    @torch.jit.ignore
    def _chunk(self, 
        a: torch.Tensor, 
        b: torch.Tensor, 
        chunk_size: int
    ) -> torch.Tensor:



        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)
        outer = torch.stack(out, dim=0)
        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:

        if mask is None:
            mask = m.new_ones(m.shape[:-1])


        m = self.layer_norm(m)


        mask = mask.unsqueeze(-1)
        a = self.linear_1(m) * mask
        b = self.linear_2(m) * mask

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)


        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)


        outer = outer / (self.eps + norm)

        return outer
