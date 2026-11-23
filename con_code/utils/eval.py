import numpy as np
import tmtools
import scipy
import MDAnalysis as mda
import numpy as np
import torch
from con_code.common.residue_con import atom_types,resatoms,restype_name_to_atom14_names, chi_angles_atoms,three_to_one,three2idx,three2self,chi_angles_atoms,amino_acid_params,rigid_group_atom_positions
from con_code.utils.structure_cal import obtain_dihediral_angles,_get_torsion,cos_sin_to_degrees,rotate_around_axis
from Bio.PDB import PDBParser, Superimposer, is_aa, Select, NeighborSearch

import os
import tmtools
import pyrosetta
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from Bio.PDB import PDBParser, is_aa
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
import numpy as np
import mdtraj as md
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def paraser_proteindic(path,chain = None):
    protein_dic = {}
    u = mda.Universe(path,topology_format='PDB') 
    if chain is not None:
        protein = u.select_atoms(f'protein and chainid {chain}')
    else:
        protein = u.select_atoms('protein')
    for ch in u.segments:
        p_ca,p_n,p_c = [],[],[]
        resls = list(ch.residues)
        phi_list=[]
        pure_res_lis, seq, node_s, X_ca, X_n, X_c = [], [], [], [], [], []
        resid = []
        segid = []
        angls =[]
        chi_angs = []
        chi_masks = []
        atoms37 =[]
        only_mainchain =[]
        for i, res in enumerate(resls):

            res_id = res.resid  
            res_name = res.resname.strip()
            if res_name not in three_to_one.keys():
                        continue
            if 'N' in res.atoms.names:
                N = res.atoms.select_atoms('name N')[0].position
            else:
                    N = res.atoms.center_of_mass
            if 'CA' in res.atoms.names:
                CA = res.atoms.select_atoms('name CA')[0].position
            else:
                CA = res.atoms.center_of_mass
            if 'C' in res.atoms.names:
                C = res.atoms.select_atoms('name C')[0].position
            else:
                C = res.atoms.center_of_mass
            if 'O' in res.atoms.names:
                O = res.atoms.select_atoms("name O")[0].position
            else:
                O = res.atoms.center_of_mass

            pos37 = np.zeros((37,3))
            for a in res.atoms:
                if a.name in atom_types:
                        pos37[atom_types.index(a.name)]=a.position
            ang = obtain_dihediral_angles(res) 
            seg = res.segid
            seq.append(three_to_one[three2self.get(res_name, 'X')])
            X_ca.append(CA)
            X_n.append(N)
            X_c.append(C)
            angls.append(ang)
            resid.append((res.segid,res_name,res.resid))
            atoms37.append(pos37)
            chi_ang= [0,0,0,0]
            chi_mask = [0,0,0,0]
            flag= False
            atomsls = [a.name for a in res.atoms]
            for ai in resatoms[res_name]:
                if ai not in atomsls:
                        flag = True
            if flag:
                only_mainchain.append(1)
            else:
                only_mainchain.append(0)
                agnles = chi_angles_atoms[res_name]

                for agi,ag in enumerate(agnles):
                        chi_mask[agi] = 1
                        pos = [torch.tensor(pos37[atom_types.index(z)]) for z in ag]
                        for pos1 in pos:
                            if torch.equal(pos1,torch.zeros(3)):
                                print(i,res)
                                warnings.warn(f"find a atom pos might be unselected in {path}", UserWarning)
                        chi_ang[agi]=_get_torsion(*pos)
            chi_masks.append(chi_mask)
            chi_angs.append(chi_ang)
        protein_dic[ch.segid]={
            'residues':seq,
            'resid':resid,
            'angls':angls,
            'chi_angs':chi_angs,
            'chi_masks':chi_masks,
            'atoms37':atoms37,

        }
    return protein_dic

def get_fullrmsd(pos1, pos2, eps=1e-8):

    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)

    mask1 = np.linalg.norm(pos1, axis=-1) > eps
    mask2 = np.linalg.norm(pos2, axis=-1) > eps
    mask = mask1 & mask2

    if mask.sum() == 0:
        return np.nan

    diff2 = np.sum((pos1 - pos2) ** 2, axis=-1)
    return np.sqrt(np.mean(diff2[mask]))

def get_carmsd(pos1,pos2):
    pos1 = pos1[:,atom_types.index('CA')]
    pos2 = pos2[:,atom_types.index('CA')]
    return np.sqrt(np.sum((pos1 - pos2)**2)/pos1.shape[0])


def get_tmscore(pos1,pos2):
    tm_results = tmtools.tm_align(pos1, pos2, 'A'*len(pos1), 'A'*len(pos2))
    return tm_results.tm_norm_chain2
def get_insertclash(pos1,cutoff = 0.5):

    eps = 1e-6
    mask = np.linalg.norm(pos1, axis=-1) > eps
    dis = scipy.spatial.distance.cdist(pos1[mask], pos1[mask])

    dism = np.where((dis<cutoff) & (dis>0),1,0)
    interclash = np.where(dism==1)
    return interclash
def get_interclash(pos1,pos2,cutoff = 1):



    eps = 1e-6
    pos1= pos1.reshape(-1,3)
    mask1= np.linalg.norm(pos1, axis=-1) > eps
    pos2= pos2.reshape(-1,3)
    mask2= np.linalg.norm(pos2, axis=-1) > eps

    dis = scipy.spatial.distance.cdist(pos1[mask1], pos2[mask2])

    dism = np.where((dis<cutoff) & (dis>0),1,0)
    interclash = np.where(dism==1)
    return interclash
def get_AAR(res1,res2):
    return (res1 == res2).sum()/len(res1)
def get_reschi(dic1,dic2):
    resdic = {}
    seq =  dic2['residues']
    chi1 =  dic1['chi_angs']
    chi2 = dic2['chi_angs']
    for i in range(len(seq)):

        diff =   (np.rad2deg(np.asarray(chi1[i]) - np.asarray(chi2[i])) + 180) % 360 - 180
        result = np.abs(diff)
        if seq[i] not in resdic.keys():
            resdic[seq[i]] = result
        else:
            resdic[seq[i]] = np.concatenate([resdic[seq[i]],result])
    for k in resdic.keys():
        resdic[k] = resdic[k].reshape(-1,4)
    return resdic
def get_chierror(resdic):
    reschierro = {}
    for k in resdic.keys():
        reschierro[k] = np.mean(np.abs(resdic[k]))
    reschierro = np.concatenate(reschierro)
    return reschierro
def get_second_stru(pdb,chain):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    chain2id = {chain.id:i for i,chain in enumerate(structure)}
    traj = md.load(pdb)
    chain_indices = traj.topology.select(f"chainid {chain2id[chain]}")
    traj = traj.atom_slice(chain_indices)
    return md.compute_dssp(traj,simplified=True)
def get_bind_site(pdb,chain_id):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    peps = [atom for res in structure[chain_id] for atom in res if atom.get_name() == 'CA']
    recs = [atom for chain in structure if chain.get_id()!=chain_id for res in chain for atom in res if atom.get_name() == 'CA']
    search = NeighborSearch(recs)
    near_res = []
    for atom in peps:
        near_res += search.search(atom.get_coord(), 10.0, level='R')
    near_res = set([res.get_id()[1] for res in near_res])
    return near_res

def get_bind_ratio(pdb1, pdb2, chain_id1, chain_id2):
    near_res1,near_res2 = get_bind_site(pdb1,chain_id1),get_bind_site(pdb2,chain_id2)
    return len(near_res1.intersection(near_res2))/(len(near_res2)+1e-10) 

def batch_calculate_angles_torch(points_A,points_B,points_C):
    vec_BA = points_A - points_B  
    vec_BC = points_C - points_B   
    norm_BA = torch.norm(vec_BA, dim=-1) + 1e-8  
    norm_BC = torch.norm(vec_BC, dim=-1) + 1e-8  
    dot_product = torch.sum(vec_BA * vec_BC, dim=-1)  

    cos_theta = dot_product / (norm_BA * norm_BC)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-8, 1.0 - 1e-8)
    theta_rad = torch.acos(cos_theta)

    return theta_rad
def get_pep_bond_metric(pos):
    CA = pos[:, :, atom_types.index('CA')]
    N = pos[:, :, atom_types.index('N')]
    C = pos[:, :, atom_types.index('C')]
    O = pos[:, :, atom_types.index('O')]

    C_bond_N = torch.norm(C[:,:-1,:]-N[:,1:,:], dim=-1)
    Ca_dis_CA = torch.norm(CA[:,:-1,:]-CA[:,1:,:], dim=-1)
    CA_C_N = batch_calculate_angles_torch(CA[:,:-1],C[:,:-1],N[:,1:])
    C_N_CA = batch_calculate_angles_torch(C[:,:-1],N[:,1:],CA[:,1:])
    O_dis_O = torch.norm(O[:,:-1,:]-O[:,1:,:], dim=-1)
    return C_bond_N,Ca_dis_CA,CA_C_N,C_N_CA,O_dis_O
def get_dis_eval(refpath,
            targetpath,refchain= None,targetchain= None ):
    dics = {}
    if refpath is not None:
        refs = paraser_proteindic(refpath)

    targets = paraser_proteindic(targetpath)

    refchainid= list(refs.keys())
    targetchainid= list(targets.keys())
    refpair=[]
    if refchain is not None:
        for i in refchainid:
            for k in targetchainid:
                if len(targets[k]['residues'])==len(refs[i]['residues']):
                    refpair.append((i,k))
        refchain,targetchain = refpair[0]
        for i,k in refpair[1:]:
            if len(targets[i]['residues'])<len(targets[targetchain]['residues']):
                refchain,targetchain = i,k
    refpair = [(refchain,targetchain)]
    for i,k in refpair:
        ref_C_bond_N,ref_Ca_dis_CA,ref_CA_C_N,ref_C_N_CA,ref_O_dis_O = get_pep_bond_metric(torch.from_numpy(np.stack(refs[i]['atoms37'],axis=0)).unsqueeze(0))
        target_C_bond_N,target_Ca_dis_CA,target_CA_C_N,target_C_N_CA,target_O_dis_O = get_pep_bond_metric(torch.from_numpy(np.stack(targets[k]['atoms37'],axis=0)).unsqueeze(0))
    for i,k in refpair:
        residues = targets[k]['residues']
        refphi = torch.from_numpy(np.array(refs[i]['angls']))[:,0]
        refpsi = torch.from_numpy(np.array(refs[i]['angls']))[:,1]  
        targetphi = torch.from_numpy(np.array(targets[k]['angls']))[:,0]
        targetpsi = torch.from_numpy(np.array(targets[k]['angls']))[:,1]

    dics= {}
    dics['residues'] = residues
    dics['targetphi'] = targetphi
    dics['targetpsi'] = targetpsi
    dics['target_C_bond_N'] = target_C_bond_N
    dics['target_Ca_dis_CA'] = target_Ca_dis_CA
    dics['target_CA_C_N'] = target_CA_C_N
    dics['target_C_N_CA'] = target_C_N_CA
    dics['refphi'] = refphi
    dics['refpsi'] = refpsi
    dics['ref_C_bond_N'] = ref_C_bond_N
    dics['ref_Ca_dis_CA'] = ref_Ca_dis_CA
    dics['ref_CA_C_N'] = ref_CA_C_N
    dics['ref_C_N_CA'] = ref_C_N_CA
    dics['ref_O_dis_O'] = ref_O_dis_O
    dics['target_O_dis_O'] = target_O_dis_O
    return dics


def complete_linkage_clusters_from_similarity(sim_mat, sim_thresh):

    sim_mat = np.asarray(sim_mat, dtype=float)
    n = sim_mat.shape[0]

    if n == 0:
        return 0, np.array([], dtype=int)

    if n == 1:
        return 1, np.array([0], dtype=int)

    sim_mat = (sim_mat + sim_mat.T) / 2
    np.fill_diagonal(sim_mat, 1.0)

    dist_mat = 1.0 - sim_mat
    np.fill_diagonal(dist_mat, 0.0)

    condensed_dist = squareform(dist_mat, checks=False)

    Z = linkage(condensed_dist, method="complete")

    raw_labels = fcluster(
        Z,
        t=1.0 - sim_thresh,
        criterion="distance",
    )

    labels = raw_labels - 1
    cluster_num = len(np.unique(labels))

    return cluster_num, labels
def get_diversity(target_paths,chain_id,tm_thresh = 0.5,aa_thresh = 0.7 ):
    targets  =[]
    for targetpath in target_paths:
        targets.append(paraser_proteindic(targetpath))
    n = len(targets)
    tm_mat = np.zeros((n, n))
    AA_mat = np.zeros((n, n))
    for i in range(n):
        targetcai = np.stack(targets[i][chain_id]['atoms37'])[:,atom_types.index('CA')]
        aa_i = np.stack(targets[i][chain_id]['residues'])
        for j in range(i+1, n):

            targetcaj = np.stack(targets[j][chain_id]['atoms37'])[:,atom_types.index('CA')]
            aa_j = np.stack(targets[j][chain_id]['residues'])
            tm = get_tmscore(targetcai,targetcaj)
            tm_mat[i,j] = tm
            tm_mat[j,i] = tm
            AAR = get_AAR(aa_i,aa_j)
            AA_mat[i,j] = AAR
            AA_mat[j,i] = AAR

    avg_tm = tm_mat[np.triu_indices(n, k=1)].mean()
    avg_AAR = AA_mat[np.triu_indices(n, k=1)].mean()
    tm_cluster_num, tm_cluster_labels = complete_linkage_clusters_from_similarity(
        tm_mat,
        tm_thresh,
    )

    aa_cluster_num, aa_cluster_labels = complete_linkage_clusters_from_similarity(
        AA_mat,
        aa_thresh,
    )

    struct_diversity=  tm_cluster_num / n
    seq_diversity=  aa_cluster_num / n
    return struct_diversity,seq_diversity,avg_tm,avg_AAR
def get_chain_dic(input_pdb):
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb)
    chain_dic = {}
    for model in structure:
        for chain in model:
            chain_dic[chain.id] = len([res for res in chain if is_aa(res) and res.has_id('CA')])

    return chain_dic




class RosettaInterfaceScorer:


    def __init__(
        self,
        chain_id="A",
        n_relax=1,
        pack_separated=True,
        pyrosetta_options="-mute all",
    ):
        self.chain_id = chain_id
        self.n_relax = n_relax
        self.pack_separated = pack_separated

        if not pyrosetta.rosetta.basic.was_init_called():
            pyrosetta.init(pyrosetta_options)

        self.scorefxn = get_fa_scorefxn()

        self.fast_relax = FastRelax()
        self.fast_relax.set_scorefxn(self.scorefxn)

        self._mover_cache = {}
    @staticmethod
    def _get_chains_from_pdb(pdb_path):
        chains = []
        seen = set()

        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    chain = line[21].strip()
                    if chain and chain not in seen:
                        seen.add(chain)
                        chains.append(chain)

        return chains

    def _get_interface(self, pdb_path):
        chains = self._get_chains_from_pdb(pdb_path)

        if self.chain_id not in chains:
            raise ValueError(
                f"chain_id={self.chain_id!r} not found in {pdb_path}. "
                f"Available chains: {chains}"
            )

        other_chains = [c for c in chains if c != self.chain_id]

        if not other_chains:
            raise ValueError(
                f"Only one chain found in {pdb_path}: {chains}. "
                "Interface scoring needs at least two chains."
            )

        return f'{self.chain_id}_{"".join(other_chains)}'

    def _get_mover(self, interface):
        key = (interface, self.pack_separated)

        if key not in self._mover_cache:
            mover = InterfaceAnalyzerMover(interface)
            mover.set_pack_separated(self.pack_separated)
            self._mover_cache[key] = mover

        return self._mover_cache[key]

    @staticmethod
    def _make_save_path(pdb_path):
        pdb_path = Path(pdb_path)

        parent = pdb_path.parent
        relaxed_parent = parent.with_name(parent.name + "_relax")
        relaxed_parent.mkdir(parents=True, exist_ok=True)

        return relaxed_parent / pdb_path.name

    def relax_in_place(self, pdb_path):
        path = Path(pdb_path)
        pose = pose_from_pdb(str(path))
        self.fast_relax.apply(pose)
        with TemporaryDirectory(dir=path.parent) as directory:
            staged = Path(directory) / path.name
            pose.dump_pdb(str(staged))
            if staged.stat().st_size == 0:
                raise RuntimeError(f'Relax produced an empty PDB for {path}')
            staged.replace(path)

    def score(self, pdb_path, save=False):
        pdb_path = str(pdb_path)

        pose0 = pose_from_pdb(pdb_path)
        interface = self._get_interface(pdb_path)
        mover = self._get_mover(interface)

        n_runs = max(1, self.n_relax)

        stabs = []
        binds = []

        best_pose = None
        best_stab = float("inf")

        for _ in range(n_runs):
            pose = pose0.clone()

            if self.n_relax > 0:
                self.fast_relax.apply(pose)

            stab = float(self.scorefxn(pose))

            ia_pose = pose.clone()
            mover.apply(ia_pose)

            bind = float(ia_pose.scores["dG_separated"])

            stabs.append(stab)
            binds.append(bind)

            if stab < best_stab:
                best_stab = stab
                best_pose = pose.clone()

        if save and best_pose is not None:
            save_path = self._make_save_path(pdb_path)
            best_pose.dump_pdb(str(save_path))

        return {
            "name": pdb_path,
            "stab": mean(stabs),
            "bind": mean(binds),
        }

    def score_many(self, pdb_paths, save=False):
        results = []

        for pdb_path in pdb_paths:
            try:
                result = self.score(pdb_path, save=save)
                results.append(result)
            except Exception as e:
                results.append({
                    "name": str(pdb_path),
                    "stab": None,
                    "bind": None,
                    "error": repr(e),
                })

        return results