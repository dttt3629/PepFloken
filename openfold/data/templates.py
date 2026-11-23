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


import dataclasses
import datetime
import glob
import json
import logging
import os
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from openfold.data import parsers, mmcif_parsing
from openfold.data.errors import Error
from openfold.data.tools import kalign
from openfold.data.tools.utils import to_date
from openfold.np import residue_constants


class NoChainsError(Error):
    pass


class SequenceNotInTemplateError(Error):
    pass


class NoAtomDataInTemplateError(Error):
    pass


class TemplateAtomMaskAllZerosError(Error):
    pass


class QueryToTemplateAlignError(Error):
    pass


class CaDistanceError(Error):
    pass



class PrefilterError(Exception):
    pass


class DateError(PrefilterError):
    pass


class PdbIdError(PrefilterError):
    pass


class AlignRatioError(PrefilterError):
    pass


class DuplicateError(PrefilterError):
    pass


class LengthError(PrefilterError):
    pass


TEMPLATE_FEATURES = {
    "template_aatype": np.int64,
    "template_all_atom_mask": np.float32,
    "template_all_atom_positions": np.float32,
    "template_domain_names": np.object,
    "template_sequence": np.object,
    "template_sum_probs": np.float32,
}


def _get_pdb_id_and_chain(hit: parsers.TemplateHit) -> Tuple[str, str]:


    id_match = re.match(r"[a-zA-Z\d]{4}_[a-zA-Z0-9.]+", hit.name)
    if not id_match:
        raise ValueError(f"hit.name did not start with PDBID_chain: {hit.name}")
    pdb_id, chain_id = id_match.group(0).split("_")
    return pdb_id.lower(), chain_id


def _is_after_cutoff(
    pdb_id: str,
    release_dates: Mapping[str, datetime.datetime],
    release_date_cutoff: Optional[datetime.datetime],
) -> bool:

    pdb_id_upper = pdb_id.upper()
    if release_date_cutoff is None:
        raise ValueError("The release_date_cutoff must not be None.")
    if pdb_id_upper in release_dates:
        return release_dates[pdb_id_upper] > release_date_cutoff
    else:


        logging.info(
            "Template structure not in release dates dict: %s", pdb_id
        )
        return False


def _parse_obsolete(obsolete_file_path: str) -> Mapping[str, str]:

    with open(obsolete_file_path) as f:
        result = {}
        for line in f:
            line = line.strip()

            if line.startswith("OBSLTE") and len(line) > 30:


                from_id = line[20:24].lower()
                to_id = line[29:33].lower()
                result[from_id] = to_id
        return result


def generate_release_dates_cache(mmcif_dir: str, out_path: str):
    dates = {}
    for f in os.listdir(mmcif_dir):
        if f.endswith(".cif"):
            path = os.path.join(mmcif_dir, f)
            with open(path, "r") as fp:
                mmcif_string = fp.read()

            file_id = os.path.splitext(f)[0]
            mmcif = mmcif_parsing.parse(
                file_id=file_id, mmcif_string=mmcif_string
            )
            if mmcif.mmcif_object is None:
                logging.info(f"Failed to parse {f}. Skipping...")
                continue

            mmcif = mmcif.mmcif_object
            release_date = mmcif.header["release_date"]

            dates[file_id] = release_date

    with open(out_path, "r") as fp:
        fp.write(json.dumps(dates))


def _parse_release_dates(path: str) -> Mapping[str, datetime.datetime]:

    with open(path, "r") as fp:
        data = json.load(fp)

    return {
        pdb.upper(): to_date(v)
        for pdb, d in data.items()
        for k, v in d.items()
        if k == "release_date"
    }


def _assess_hhsearch_hit(
    hit: parsers.TemplateHit,
    hit_pdb_code: str,
    query_sequence: str,
    query_pdb_code: Optional[str],
    release_dates: Mapping[str, datetime.datetime],
    release_date_cutoff: datetime.datetime,
    max_subsequence_ratio: float = 0.95,
    min_align_ratio: float = 0.1,
) -> bool:

    aligned_cols = hit.aligned_cols
    align_ratio = aligned_cols / len(query_sequence)

    template_sequence = hit.hit_sequence.replace("-", "")
    length_ratio = float(len(template_sequence)) / len(query_sequence)



    duplicate = (
        template_sequence in query_sequence
        and length_ratio > max_subsequence_ratio
    )

    if _is_after_cutoff(hit_pdb_code, release_dates, release_date_cutoff):
        date = release_dates[hit_pdb_code.upper()]
        raise DateError(
            f"Date ({date}) > max template date "
            f"({release_date_cutoff})."
        )

    if query_pdb_code is not None:
        if query_pdb_code.lower() == hit_pdb_code.lower():
            raise PdbIdError("PDB code identical to Query PDB code.")

    if align_ratio <= min_align_ratio:
        raise AlignRatioError(
            "Proportion of residues aligned to query too small. "
            f"Align ratio: {align_ratio}."
        )

    if duplicate:
        raise DuplicateError(
            "Template is an exact subsequence of query with large "
            f"coverage. Length ratio: {length_ratio}."
        )

    if len(template_sequence) < 10:
        raise LengthError(
            f"Template too short. Length: {len(template_sequence)}."
        )

    return True


def _find_template_in_pdb(
    template_chain_id: str,
    template_sequence: str,
    mmcif_object: mmcif_parsing.MmcifObject,
) -> Tuple[str, str, int]:


    pdb_id = mmcif_object.file_id
    chain_sequence = mmcif_object.chain_to_seqres.get(template_chain_id)
    if chain_sequence and (template_sequence in chain_sequence):
        logging.info(
            "Found an exact template match %s_%s.", pdb_id, template_chain_id
        )
        mapping_offset = chain_sequence.find(template_sequence)
        return chain_sequence, template_chain_id, mapping_offset


    for chain_id, chain_sequence in mmcif_object.chain_to_seqres.items():
        if chain_sequence and (template_sequence in chain_sequence):
            logging.info("Found a sequence-only match %s_%s.", pdb_id, chain_id)
            mapping_offset = chain_sequence.find(template_sequence)
            return chain_sequence, chain_id, mapping_offset



    regex = ["." if aa == "X" else "(?:%s|X)" % aa for aa in template_sequence]
    regex = re.compile("".join(regex))
    for chain_id, chain_sequence in mmcif_object.chain_to_seqres.items():
        match = re.search(regex, chain_sequence)
        if match:
            logging.info(
                "Found a fuzzy sequence-only match %s_%s.", pdb_id, chain_id
            )
            mapping_offset = match.start()
            return chain_sequence, chain_id, mapping_offset


    raise SequenceNotInTemplateError(
        "Could not find the template sequence in %s_%s. Template sequence: %s, "
        "chain_to_seqres: %s"
        % (
            pdb_id,
            template_chain_id,
            template_sequence,
            mmcif_object.chain_to_seqres,
        )
    )


def _realign_pdb_template_to_query(
    old_template_sequence: str,
    template_chain_id: str,
    mmcif_object: mmcif_parsing.MmcifObject,
    old_mapping: Mapping[int, int],
    kalign_binary_path: str,
) -> Tuple[str, Mapping[int, int]]:

    aligner = kalign.Kalign(binary_path=kalign_binary_path)
    new_template_sequence = mmcif_object.chain_to_seqres.get(
        template_chain_id, ""
    )



    if not new_template_sequence:
        if len(mmcif_object.chain_to_seqres) == 1:
            logging.info(
                "Could not find %s in %s, but there is only 1 sequence, so "
                "using that one.",
                template_chain_id,
                mmcif_object.file_id,
            )
            new_template_sequence = list(mmcif_object.chain_to_seqres.values())[
                0
            ]
        else:
            raise QueryToTemplateAlignError(
                f"Could not find chain {template_chain_id} in {mmcif_object.file_id}. "
                "If there are no mmCIF parsing errors, it is possible it was not a "
                "protein chain."
            )

    try:
        (old_aligned_template, new_aligned_template), _ = parsers.parse_a3m(
            aligner.align([old_template_sequence, new_template_sequence])
        )
    except Exception as e:
        raise QueryToTemplateAlignError(
            "Could not align old template %s to template %s (%s_%s). Error: %s"
            % (
                old_template_sequence,
                new_template_sequence,
                mmcif_object.file_id,
                template_chain_id,
                str(e),
            )
        )

    logging.info(
        "Old aligned template: %s\nNew aligned template: %s",
        old_aligned_template,
        new_aligned_template,
    )

    old_to_new_template_mapping = {}
    old_template_index = -1
    new_template_index = -1
    num_same = 0
    for old_template_aa, new_template_aa in zip(
        old_aligned_template, new_aligned_template
    ):
        if old_template_aa != "-":
            old_template_index += 1
        if new_template_aa != "-":
            new_template_index += 1
        if old_template_aa != "-" and new_template_aa != "-":
            old_to_new_template_mapping[old_template_index] = new_template_index
            if old_template_aa == new_template_aa:
                num_same += 1


    if (
        float(num_same)
        / min(len(old_template_sequence), len(new_template_sequence))
        < 0.9
    ):
        raise QueryToTemplateAlignError(
            "Insufficient similarity of the sequence in the database: %s to the "
            "actual sequence in the mmCIF file %s_%s: %s. We require at least "
            "90 %% similarity wrt to the shorter of the sequences. This is not a "
            "problem unless you think this is a template that should be included."
            % (
                old_template_sequence,
                mmcif_object.file_id,
                template_chain_id,
                new_template_sequence,
            )
        )

    new_query_to_template_mapping = {}
    for query_index, old_template_index in old_mapping.items():
        new_query_to_template_mapping[
            query_index
        ] = old_to_new_template_mapping.get(old_template_index, -1)

    new_template_sequence = new_template_sequence.replace("-", "")

    return new_template_sequence, new_query_to_template_mapping


def _check_residue_distances(
    all_positions: np.ndarray,
    all_positions_mask: np.ndarray,
    max_ca_ca_distance: float,
):

    ca_position = residue_constants.atom_order["CA"]
    prev_is_unmasked = False
    prev_calpha = None
    for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):
        this_is_unmasked = bool(mask[ca_position])
        if this_is_unmasked:
            this_calpha = coords[ca_position]
            if prev_is_unmasked:
                distance = np.linalg.norm(this_calpha - prev_calpha)
                if distance > max_ca_ca_distance:
                    raise CaDistanceError(
                        "The distance between residues %d and %d is %f > limit %f."
                        % (i, i + 1, distance, max_ca_ca_distance)
                    )
            prev_calpha = this_calpha
        prev_is_unmasked = this_is_unmasked


def _get_atom_positions(
    mmcif_object: mmcif_parsing.MmcifObject,
    auth_chain_id: str,
    max_ca_ca_distance: float,
    _zero_center_positions: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:

    coords_with_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, 
        chain_id=auth_chain_id,
        _zero_center_positions=_zero_center_positions,
    )
    all_atom_positions, all_atom_mask = coords_with_mask
    _check_residue_distances(
        all_atom_positions, all_atom_mask, max_ca_ca_distance
    )
    return all_atom_positions, all_atom_mask


def _extract_template_features(
    mmcif_object: mmcif_parsing.MmcifObject,
    pdb_id: str,
    mapping: Mapping[int, int],
    template_sequence: str,
    query_sequence: str,
    template_chain_id: str,
    kalign_binary_path: str,
    _zero_center_positions: bool = True,
) -> Tuple[Dict[str, Any], Optional[str]]:

    if mmcif_object is None or not mmcif_object.chain_to_seqres:
        raise NoChainsError(
            "No chains in PDB: %s_%s" % (pdb_id, template_chain_id)
        )

    warning = None
    try:
        seqres, chain_id, mapping_offset = _find_template_in_pdb(
            template_chain_id=template_chain_id,
            template_sequence=template_sequence,
            mmcif_object=mmcif_object,
        )
    except SequenceNotInTemplateError:


        chain_id = template_chain_id
        warning = (
            f"The exact sequence {template_sequence} was not found in "
            f"{pdb_id}_{chain_id}. Realigning the template to the actual sequence."
        )
        logging.warning(warning)

        seqres, mapping = _realign_pdb_template_to_query(
            old_template_sequence=template_sequence,
            template_chain_id=template_chain_id,
            mmcif_object=mmcif_object,
            old_mapping=mapping,
            kalign_binary_path=kalign_binary_path,
        )
        logging.info(
            "Sequence in %s_%s: %s successfully realigned to %s",
            pdb_id,
            chain_id,
            template_sequence,
            seqres,
        )

        template_sequence = seqres

        mapping_offset = 0

    try:


        all_atom_positions, all_atom_mask = _get_atom_positions(
            mmcif_object, 
            chain_id, 
            max_ca_ca_distance=150.0, 
            _zero_center_positions=_zero_center_positions,
        )
    except (CaDistanceError, KeyError) as ex:
        raise NoAtomDataInTemplateError(
            "Could not get atom data (%s_%s): %s" % (pdb_id, chain_id, str(ex))
        ) from ex

    all_atom_positions = np.split(
        all_atom_positions, all_atom_positions.shape[0]
    )
    all_atom_masks = np.split(all_atom_mask, all_atom_mask.shape[0])

    output_templates_sequence = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    for _ in query_sequence:

        templates_all_atom_positions.append(
            np.zeros((residue_constants.atom_type_num, 3))
        )
        templates_all_atom_masks.append(
            np.zeros(residue_constants.atom_type_num)
        )
        output_templates_sequence.append("-")

    for k, v in mapping.items():
        template_index = v + mapping_offset
        templates_all_atom_positions[k] = all_atom_positions[template_index][0]
        templates_all_atom_masks[k] = all_atom_masks[template_index][0]
        output_templates_sequence[k] = template_sequence[v]


    if np.sum(templates_all_atom_masks) < 5:
        raise TemplateAtomMaskAllZerosError(
            "Template all atom mask was all zeros: %s_%s. Residue range: %d-%d"
            % (
                pdb_id,
                chain_id,
                min(mapping.values()) + mapping_offset,
                max(mapping.values()) + mapping_offset,
            )
        )

    output_templates_sequence = "".join(output_templates_sequence)

    templates_aatype = residue_constants.sequence_to_onehot(
        output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID
    )

    return (
        {
            "template_all_atom_positions": np.array(
                templates_all_atom_positions
            ),
            "template_all_atom_mask": np.array(templates_all_atom_masks),
            "template_sequence": output_templates_sequence.encode(),
            "template_aatype": np.array(templates_aatype),
            "template_domain_names": f"{pdb_id.lower()}_{chain_id}".encode(),
        },
        warning,
    )


def _build_query_to_hit_index_mapping(
    hit_query_sequence: str,
    hit_sequence: str,
    indices_hit: Sequence[int],
    indices_query: Sequence[int],
    original_query_sequence: str,
) -> Mapping[int, int]:


    if not hit_query_sequence:
        return {}


    hhsearch_query_sequence = hit_query_sequence.replace("-", "")
    hit_sequence = hit_sequence.replace("-", "")
    hhsearch_query_offset = original_query_sequence.find(
        hhsearch_query_sequence
    )


    min_idx = min(x for x in indices_hit if x > -1)
    fixed_indices_hit = [x - min_idx if x > -1 else -1 for x in indices_hit]

    min_idx = min(x for x in indices_query if x > -1)
    fixed_indices_query = [x - min_idx if x > -1 else -1 for x in indices_query]


    mapping = {}
    for q_i, q_t in zip(fixed_indices_query, fixed_indices_hit):
        if q_t != -1 and q_i != -1:
            if q_t >= len(hit_sequence) or q_i + hhsearch_query_offset >= len(
                original_query_sequence
            ):
                continue
            mapping[q_i + hhsearch_query_offset] = q_t

    return mapping


@dataclasses.dataclass(frozen=True)
class PrefilterResult:
    valid: bool
    error: Optional[str]
    warning: Optional[str]

@dataclasses.dataclass(frozen=True)
class SingleHitResult:
    features: Optional[Mapping[str, Any]]
    error: Optional[str]
    warning: Optional[str]


def _prefilter_hit(
    query_sequence: str,
    query_pdb_code: Optional[str],
    hit: parsers.TemplateHit,
    max_template_date: datetime.datetime,
    release_dates: Mapping[str, datetime.datetime],
    obsolete_pdbs: Mapping[str, str],
    strict_error_check: bool = False,
):

    hit_pdb_code, hit_chain_id = _get_pdb_id_and_chain(hit)

    if hit_pdb_code not in release_dates:
        if hit_pdb_code in obsolete_pdbs:
            hit_pdb_code = obsolete_pdbs[hit_pdb_code]



    try:
        _assess_hhsearch_hit(
            hit=hit,
            hit_pdb_code=hit_pdb_code,
            query_sequence=query_sequence,
            query_pdb_code=query_pdb_code,
            release_dates=release_dates,
            release_date_cutoff=max_template_date,
        )
    except PrefilterError as e:
        hit_name = f"{hit_pdb_code}_{hit_chain_id}"
        msg = f"hit {hit_name} did not pass prefilter: {str(e)}"
        logging.info("%s: %s", query_pdb_code, msg)
        if strict_error_check and isinstance(
            e, (DateError, PdbIdError, DuplicateError)
        ):

            return PrefilterResult(valid=False, error=msg, warning=None)

        return PrefilterResult(valid=False, error=None, warning=None)

    return PrefilterResult(valid=True, error=None, warning=None)


def _process_single_hit(
    query_sequence: str,
    query_pdb_code: Optional[str],
    hit: parsers.TemplateHit,
    mmcif_dir: str,
    max_template_date: datetime.datetime,
    release_dates: Mapping[str, datetime.datetime],
    obsolete_pdbs: Mapping[str, str],
    kalign_binary_path: str,
    strict_error_check: bool = False,
    _zero_center_positions: bool = True,
) -> SingleHitResult:


    hit_pdb_code, hit_chain_id = _get_pdb_id_and_chain(hit)

    if hit_pdb_code not in release_dates:
        if hit_pdb_code in obsolete_pdbs:
            hit_pdb_code = obsolete_pdbs[hit_pdb_code]

    mapping = _build_query_to_hit_index_mapping(
        hit.query,
        hit.hit_sequence,
        hit.indices_hit,
        hit.indices_query,
        query_sequence,
    )



    template_sequence = hit.hit_sequence.replace("-", "")

    cif_path = os.path.join(mmcif_dir, hit_pdb_code + ".cif")
    logging.info(
        "Reading PDB entry from %s. Query: %s, template: %s",
        cif_path,
        query_sequence,
        template_sequence,
    )

    with open(cif_path, "r") as cif_file:
        cif_string = cif_file.read()

    parsing_result = mmcif_parsing.parse(
        file_id=hit_pdb_code, mmcif_string=cif_string
    )

    if parsing_result.mmcif_object is not None:
        hit_release_date = datetime.datetime.strptime(
            parsing_result.mmcif_object.header["release_date"], "%Y-%m-%d"
        )
        if hit_release_date > max_template_date:
            error = "Template %s date (%s) > max template date (%s)." % (
                hit_pdb_code,
                hit_release_date,
                max_template_date,
            )
            if strict_error_check:
                return SingleHitResult(features=None, error=error, warning=None)
            else:
                logging.info(error)
                return SingleHitResult(features=None, error=None, warning=None)

    try:
        features, realign_warning = _extract_template_features(
            mmcif_object=parsing_result.mmcif_object,
            pdb_id=hit_pdb_code,
            mapping=mapping,
            template_sequence=template_sequence,
            query_sequence=query_sequence,
            template_chain_id=hit_chain_id,
            kalign_binary_path=kalign_binary_path,
            _zero_center_positions=_zero_center_positions,
        )
        features["template_sum_probs"] = [hit.sum_probs]




        return SingleHitResult(
            features=features, error=None, warning=realign_warning
        )
    except (
        NoChainsError,
        NoAtomDataInTemplateError,
        TemplateAtomMaskAllZerosError,
    ) as e:


        warning = (
            "%s_%s (sum_probs: %.2f, rank: %d): feature extracting errors: "
            "%s, mmCIF parsing errors: %s"
            % (
                hit_pdb_code,
                hit_chain_id,
                hit.sum_probs,
                hit.index,
                str(e),
                parsing_result.errors,
            )
        )
        if strict_error_check:
            return SingleHitResult(features=None, error=warning, warning=None)
        else:
            return SingleHitResult(features=None, error=None, warning=warning)
    except Error as e:
        error = (
            "%s_%s (sum_probs: %.2f, rank: %d): feature extracting errors: "
            "%s, mmCIF parsing errors: %s"
            % (
                hit_pdb_code,
                hit_chain_id,
                hit.sum_probs,
                hit.index,
                str(e),
                parsing_result.errors,
            )
        )
        return SingleHitResult(features=None, error=error, warning=None)


@dataclasses.dataclass(frozen=True)
class TemplateSearchResult:
    features: Mapping[str, Any]
    errors: Sequence[str]
    warnings: Sequence[str]


class TemplateHitFeaturizer:

    def __init__(
        self,
        mmcif_dir: str,
        max_template_date: str,
        max_hits: int,
        kalign_binary_path: str,
        release_dates_path: Optional[str] = None,
        obsolete_pdbs_path: Optional[str] = None,
        strict_error_check: bool = False,
        _shuffle_top_k_prefiltered: Optional[int] = None,
        _zero_center_positions: bool = True,
    ):

        self._mmcif_dir = mmcif_dir
        if not glob.glob(os.path.join(self._mmcif_dir, "*.cif")):
            logging.error("Could not find CIFs in %s", self._mmcif_dir)
            raise ValueError(f"Could not find CIFs in {self._mmcif_dir}")

        try:
            self._max_template_date = datetime.datetime.strptime(
                max_template_date, "%Y-%m-%d"
            )
        except ValueError:
            raise ValueError(
                "max_template_date must be set and have format YYYY-MM-DD."
            )
        self.max_hits = max_hits
        self._kalign_binary_path = kalign_binary_path
        self._strict_error_check = strict_error_check

        if release_dates_path:
            logging.info(
                "Using precomputed release dates %s.", release_dates_path
            )
            self._release_dates = _parse_release_dates(release_dates_path)
        else:
            self._release_dates = {}

        if obsolete_pdbs_path:
            logging.info(
                "Using precomputed obsolete pdbs %s.", obsolete_pdbs_path
            )
            self._obsolete_pdbs = _parse_obsolete(obsolete_pdbs_path)
        else:
            self._obsolete_pdbs = {}

        self._shuffle_top_k_prefiltered = _shuffle_top_k_prefiltered
        self._zero_center_positions = _zero_center_positions

    def get_templates(
        self,
        query_sequence: str,
        query_pdb_code: Optional[str],
        query_release_date: Optional[datetime.datetime],
        hits: Sequence[parsers.TemplateHit],
    ) -> TemplateSearchResult:

        logging.info("Searching for template for: %s", query_pdb_code)

        template_features = {}
        for template_feature_name in TEMPLATE_FEATURES:
            template_features[template_feature_name] = []



        template_cutoff_date = self._max_template_date
        if query_release_date:
            delta = datetime.timedelta(days=60)
            if query_release_date - delta < template_cutoff_date:
                template_cutoff_date = query_release_date - delta
            assert template_cutoff_date < query_release_date
        assert template_cutoff_date <= self._max_template_date

        num_hits = 0
        errors = []
        warnings = []

        filtered = []
        for hit in hits:
            prefilter_result = _prefilter_hit(
                query_sequence=query_sequence,
                query_pdb_code=query_pdb_code,
                hit=hit,
                max_template_date=template_cutoff_date,
                release_dates=self._release_dates,
                obsolete_pdbs=self._obsolete_pdbs,
                strict_error_check=self._strict_error_check,
            )

            if prefilter_result.error:
                errors.append(prefilter_result.error)

            if prefilter_result.warning:
                warnings.append(prefilter_result.warning)

            if prefilter_result.valid:
                filtered.append(hit)

        filtered = list(
            sorted(filtered, key=lambda x: x.sum_probs, reverse=True)
        )
        idx = list(range(len(filtered)))
        if(self._shuffle_top_k_prefiltered):
            stk = self._shuffle_top_k_prefiltered
            idx[:stk] = np.random.permutation(idx[:stk])

        for i in idx:

            if num_hits >= self.max_hits:
                break

            hit = filtered[i]

            result = _process_single_hit(
                query_sequence=query_sequence,
                query_pdb_code=query_pdb_code,
                hit=hit,
                mmcif_dir=self._mmcif_dir,
                max_template_date=template_cutoff_date,
                release_dates=self._release_dates,
                obsolete_pdbs=self._obsolete_pdbs,
                strict_error_check=self._strict_error_check,
                kalign_binary_path=self._kalign_binary_path,
                _zero_center_positions=self._zero_center_positions,
            )

            if result.error:
                errors.append(result.error)



            if result.warning:
                warnings.append(result.warning)

            if result.features is None:
                logging.info(
                    "Skipped invalid hit %s, error: %s, warning: %s",
                    hit.name,
                    result.error,
                    result.warning,
                )
            else:

                num_hits += 1
                for k in template_features:
                    template_features[k].append(result.features[k])

        for name in template_features:
            if num_hits > 0:
                template_features[name] = np.stack(
                    template_features[name], axis=0
                ).astype(TEMPLATE_FEATURES[name])
            else:

                template_features[name] = np.array(
                    [], dtype=TEMPLATE_FEATURES[name]
                )

        return TemplateSearchResult(
            features=template_features, errors=errors, warnings=warnings
        )
