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

import pdbfixer
from simtk.openmm import app
from simtk.openmm.app import element


def fix_pdb(pdbfile, alterations_info):

    fixer = pdbfixer.PDBFixer(pdbfile=pdbfile)
    fixer.findNonstandardResidues()
    alterations_info["nonstandard_residues"] = fixer.nonstandardResidues
    fixer.replaceNonstandardResidues()
    _remove_heterogens(fixer, alterations_info, keep_water=False)
    fixer.findMissingResidues()
    alterations_info["missing_residues"] = fixer.missingResidues
    fixer.findMissingAtoms()
    alterations_info["missing_heavy_atoms"] = fixer.missingAtoms
    alterations_info["missing_terminals"] = fixer.missingTerminals
    fixer.addMissingAtoms(seed=0)
    fixer.addMissingHydrogens()
    out_handle = io.StringIO()
    app.PDBFile.writeFile(
        fixer.topology, fixer.positions, out_handle, keepIds=True
    )
    return out_handle.getvalue()


def clean_structure(pdb_structure, alterations_info):

    _replace_met_se(pdb_structure, alterations_info)
    _remove_chains_of_length_one(pdb_structure, alterations_info)


def _remove_heterogens(fixer, alterations_info, keep_water):

    initial_resnames = set()
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            initial_resnames.add(residue.name)
    fixer.removeHeterogens(keepWater=keep_water)
    final_resnames = set()
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            final_resnames.add(residue.name)
    alterations_info["removed_heterogens"] = initial_resnames.difference(
        final_resnames
    )


def _replace_met_se(pdb_structure, alterations_info):

    modified_met_residues = []
    for res in pdb_structure.iter_residues():
        name = res.get_name_with_spaces().strip()
        if name == "MET":
            s_atom = res.get_atom("SD")
            if s_atom.element_symbol == "Se":
                s_atom.element_symbol = "S"
                s_atom.element = element.get_by_symbol("S")
                modified_met_residues.append(s_atom.residue_number)
    alterations_info["Se_in_MET"] = modified_met_residues


def _remove_chains_of_length_one(pdb_structure, alterations_info):

    removed_chains = {}
    for model in pdb_structure.iter_models():
        valid_chains = [c for c in model.iter_chains() if len(c) > 1]
        invalid_chain_ids = [
            c.chain_id for c in model.iter_chains() if len(c) <= 1
        ]
        model.chains = valid_chains
        for chain_id in invalid_chain_ids:
            model.chains_by_id.pop(chain_id)
        removed_chains[model.number] = invalid_chain_ids
    alterations_info["removed_chains"] = removed_chains
