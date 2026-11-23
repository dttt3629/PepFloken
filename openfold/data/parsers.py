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


import collections
import dataclasses
import re
import string
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DeletionMatrix = Sequence[Sequence[int]]


@dataclasses.dataclass(frozen=True)
class TemplateHit:


    index: int
    name: str
    aligned_cols: int
    sum_probs: float
    query: str
    hit_sequence: str
    indices_query: List[int]
    indices_hit: List[int]


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:

    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])
            sequences.append("")
            continue
        elif not line:
            continue
        sequences[index] += line

    return sequences, descriptions


def parse_stockholm(
    stockholm_string: str,
) -> Tuple[Sequence[str], DeletionMatrix, Sequence[str]]:

    name_to_sequence = collections.OrderedDict()
    for line in stockholm_string.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        name, sequence = line.split()
        if name not in name_to_sequence:
            name_to_sequence[name] = ""
        name_to_sequence[name] += sequence

    msa = []
    deletion_matrix = []

    query = ""
    keep_columns = []
    for seq_index, sequence in enumerate(name_to_sequence.values()):
        if seq_index == 0:

            query = sequence
            keep_columns = [i for i, res in enumerate(query) if res != "-"]


        aligned_sequence = "".join([sequence[c] for c in keep_columns])

        msa.append(aligned_sequence)


        deletion_vec = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query):
            if seq_res != "-" or query_res != "-":
                if query_res == "-":
                    deletion_count += 1
                else:
                    deletion_vec.append(deletion_count)
                    deletion_count = 0
        deletion_matrix.append(deletion_vec)

    return msa, deletion_matrix, list(name_to_sequence.keys())


def parse_a3m(a3m_string: str) -> Tuple[Sequence[str], DeletionMatrix]:

    sequences, _ = parse_fasta(a3m_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)


    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return aligned_sequences, deletion_matrix


def _convert_sto_seq_to_a3m(
    query_non_gaps: Sequence[bool], sto_seq: str
) -> Iterable[str]:
    for is_query_res_non_gap, sequence_res in zip(query_non_gaps, sto_seq):
        if is_query_res_non_gap:
            yield sequence_res
        elif sequence_res != "-":
            yield sequence_res.lower()


def convert_stockholm_to_a3m(
    stockholm_format: str, max_sequences: Optional[int] = None
) -> str:

    descriptions = {}
    sequences = {}
    reached_max_sequences = False

    for line in stockholm_format.splitlines():
        reached_max_sequences = (
            max_sequences and len(sequences) >= max_sequences
        )
        if line.strip() and not line.startswith(("#", "//")):


            seqname, aligned_seq = line.split(maxsplit=1)
            if seqname not in sequences:
                if reached_max_sequences:
                    continue
                sequences[seqname] = ""
            sequences[seqname] += aligned_seq

    for line in stockholm_format.splitlines():
        if line[:4] == "#=GS":


            columns = line.split(maxsplit=3)
            seqname, feature = columns[1:3]
            value = columns[3] if len(columns) == 4 else ""
            if feature != "DE":
                continue
            if reached_max_sequences and seqname not in sequences:
                continue
            descriptions[seqname] = value
            if len(descriptions) == len(sequences):
                break


    a3m_sequences = {}

    query_sequence = next(iter(sequences.values()))
    query_non_gaps = [res != "-" for res in query_sequence]
    for seqname, sto_sequence in sequences.items():
        a3m_sequences[seqname] = "".join(
            _convert_sto_seq_to_a3m(query_non_gaps, sto_sequence)
        )

    fasta_chunks = (
        f">{k} {descriptions.get(k, '')}\n{a3m_sequences[k]}"
        for k in a3m_sequences
    )
    return "\n".join(fasta_chunks) + "\n"


def _get_hhr_line_regex_groups(
    regex_pattern: str, line: str
) -> Sequence[Optional[str]]:
    match = re.match(regex_pattern, line)
    if match is None:
        raise RuntimeError(f"Could not parse query line {line}")
    return match.groups()


def _update_hhr_residue_indices_list(
    sequence: str, start_index: int, indices_list: List[int]
):

    counter = start_index
    for symbol in sequence:
        if symbol == "-":
            indices_list.append(-1)
        else:
            indices_list.append(counter)
            counter += 1


def _parse_hhr_hit(detailed_lines: Sequence[str]) -> TemplateHit:


    number_of_hit = int(detailed_lines[0].split()[-1])
    name_hit = detailed_lines[1][1:]


    pattern = (
        "Probab=(.*)[\t ]*E-value=(.*)[\t ]*Score=(.*)[\t ]*Aligned_cols=(.*)[\t"
        " ]*Identities=(.*)%[\t ]*Similarity=(.*)[\t ]*Sum_probs=(.*)[\t "
        "]*Template_Neff=(.*)"
    )
    match = re.match(pattern, detailed_lines[2])
    if match is None:
        raise RuntimeError(
            "Could not parse section: %s. Expected this: \n%s to contain summary."
            % (detailed_lines, detailed_lines[2])
        )
    (prob_true, e_value, _, aligned_cols, _, _, sum_probs, neff) = [
        float(x) for x in match.groups()
    ]





    query = ""
    hit_sequence = ""
    indices_query = []
    indices_hit = []
    length_block = None

    for line in detailed_lines[3:]:

        if (
            line.startswith("Q ")
            and not line.startswith("Q ss_dssp")
            and not line.startswith("Q ss_pred")
            and not line.startswith("Q Consensus")
        ):



            patt = r"[\t ]*([0-9]*) ([A-Z-]*)[\t ]*([0-9]*) \([0-9]*\)"
            groups = _get_hhr_line_regex_groups(patt, line[17:])



            start = int(groups[0]) - 1
            delta_query = groups[1]
            end = int(groups[2])
            num_insertions = len([x for x in delta_query if x == "-"])
            length_block = end - start + num_insertions
            assert length_block == len(delta_query)


            query += delta_query
            _update_hhr_residue_indices_list(delta_query, start, indices_query)

        elif line.startswith("T "):

            if (
                not line.startswith("T ss_dssp")
                and not line.startswith("T ss_pred")
                and not line.startswith("T Consensus")
            ):



                patt = r"[\t ]*([0-9]*) ([A-Z-]*)[\t ]*[0-9]* \([0-9]*\)"
                groups = _get_hhr_line_regex_groups(patt, line[17:])
                start = int(groups[0]) - 1
                delta_hit_sequence = groups[1]
                assert length_block == len(delta_hit_sequence)


                hit_sequence += delta_hit_sequence
                _update_hhr_residue_indices_list(
                    delta_hit_sequence, start, indices_hit
                )

    return TemplateHit(
        index=number_of_hit,
        name=name_hit,
        aligned_cols=int(aligned_cols),
        sum_probs=sum_probs,
        query=query,
        hit_sequence=hit_sequence,
        indices_query=indices_query,
        indices_hit=indices_hit,
    )


def parse_hhr(hhr_string: str) -> Sequence[TemplateHit]:

    lines = hhr_string.splitlines()





    block_starts = [i for i, line in enumerate(lines) if line.startswith("No ")]

    hits = []
    if block_starts:
        block_starts.append(len(lines))
        for i in range(len(block_starts) - 1):
            hits.append(
                _parse_hhr_hit(lines[block_starts[i] : block_starts[i + 1]])
            )
    return hits


def parse_e_values_from_tblout(tblout: str) -> Dict[str, float]:

    e_values = {"query": 0}
    lines = [line for line in tblout.splitlines() if line[0] != "#"]



    for line in lines:
        fields = line.split()
        e_value = fields[4]
        target_name = fields[0]
        e_values[target_name] = float(e_value)
    return e_values
