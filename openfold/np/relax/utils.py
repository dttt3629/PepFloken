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


import io
from openfold.np import residue_constants
from Bio import PDB
import numpy as np


from openmm import app as openmm_app
from openmm.app.internal.pdbstructure import PdbStructure


def overwrite_pdb_coordinates(pdb_str: str, pos) -> str:
    pdb_file = io.StringIO(pdb_str)
    structure = PdbStructure(pdb_file)
    topology = openmm_app.PDBFile(structure).getTopology()
    with io.StringIO() as f:
        openmm_app.PDBFile.writeFile(topology, pos, f)
        return f.getvalue()


def overwrite_b_factors(pdb_str: str, bfactors: np.ndarray) -> str:

    if bfactors.shape[-1] != residue_constants.atom_type_num:
        raise ValueError(
            f"Invalid final dimension size for bfactors: {bfactors.shape[-1]}."
        )

    parser = PDB.PDBParser(QUIET=True)
    handle = io.StringIO(pdb_str)
    structure = parser.get_structure("", handle)

    curr_resid = ("", "", "")
    idx = -1
    for atom in structure.get_atoms():
        atom_resid = atom.parent.get_id()
        if atom_resid != curr_resid:
            idx += 1
            if idx >= bfactors.shape[0]:
                raise ValueError(
                    "Index into bfactors exceeds number of residues. "
                    "B-factors shape: {shape}, idx: {idx}."
                )
        curr_resid = atom_resid
        atom.bfactor = bfactors[idx, residue_constants.atom_order["CA"]]

    new_pdb = io.StringIO()
    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(new_pdb)
    return new_pdb.getvalue()


def assert_equal_nonterminal_atom_types(
    atom_mask: np.ndarray, ref_atom_mask: np.ndarray
):


    oxt = residue_constants.atom_order["OXT"]
    no_oxt_mask = np.ones(shape=atom_mask.shape, dtype=np.bool)
    no_oxt_mask[..., oxt] = False
    np.testing.assert_almost_equal(
        ref_atom_mask[no_oxt_mask], atom_mask[no_oxt_mask]
    )
