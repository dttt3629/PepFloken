import numpy as np
from math import sin, cos, radians
from con_code.common.residue_con import atom_types,resatoms,restype_name_to_atom14_names, chi_angles_atoms,three_to_one,three2idx,three2self
import torch
from scipy.spatial.transform import Rotation 
from numpy.linalg import norm, inv
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.transformations import rotation_matrix
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm, inv, qr, det
def solve_target_vector_with_norm(n_coord, ca_coord, c_coord,o_coord, theta1=109.5, theta2=109.5, c_norm=1.54):

    N = np.array(n_coord, dtype=np.float64).reshape(-1,3)
    CA = np.array(ca_coord, dtype=np.float64).reshape(-1,3)
    C = np.array(c_coord, dtype=np.float64).reshape(-1,3)
    O = np.array(o_coord, dtype=np.float64).reshape(-1,3)
    a = N - CA  
    b = C - CA  
    a_unit = a / np.linalg.norm(a,axis=1,keepdims=True)
    b_unit = b / np.linalg.norm(b,axis=1,keepdims=True)
    cross_ab = np.cross(a_unit, b_unit)

    x_prime = a_unit
    proj_b_on_a = np.sum(b_unit * x_prime, axis=1, keepdims=True) * x_prime
    y_prime = (b_unit - proj_b_on_a) / np.linalg.norm(b_unit - proj_b_on_a,axis=1,keepdims=True)
    z_prime = cross_ab / np.linalg.norm(cross_ab,axis=1,keepdims=True)

    x = c_norm * np.cos(np.deg2rad(theta1))  
    b_x_prime = np.sum(b_unit * x_prime, axis=1, keepdims=True)
    b_y_prime = np.sum(b_unit * y_prime, axis=1, keepdims=True)
    y = (c_norm * np.cos(np.deg2rad(theta2)) - x * b_x_prime) / b_y_prime
    z_squared = c_norm**2 - x**2 - y**2

    z = np.sqrt(np.maximum(z_squared, 0.0))

    c1 = CA + x * x_prime + y * y_prime + z * z_prime
    c2 = CA + x * x_prime + y * y_prime - z * z_prime
    vec_c_o = O - C  
    vec_ca_cb1 = c1 - CA
    vec_ca_cb2 = c2 - CA
    cross_nc = np.cross(a, b)
    cross_nc_norm = cross_nc / np.linalg.norm(cross_nc,axis=1,keepdims=True)  
    vec_ca_cb1_norm = vec_ca_cb1 / np.linalg.norm(vec_ca_cb1,axis=1,keepdims=True)
    vec_ca_cb2_norm = vec_ca_cb2 / np.linalg.norm(vec_ca_cb2,axis=1,keepdims=True)
    norm_cb1 = np.linalg.norm(vec_ca_cb1,axis=1,keepdims=True)
    norm_cb2 = np.linalg.norm(vec_ca_cb2,axis=1,keepdims=True)
    vec_c_o_norm = vec_c_o / np.linalg.norm(vec_c_o, axis=1, keepdims=True)
    ang_cb1 = np.sum(vec_ca_cb1_norm * vec_c_o_norm, axis=1, keepdims=True)
    ang_cb2 = np.sum(vec_ca_cb2_norm * vec_c_o_norm, axis=1, keepdims=True)
    dot1 = np.sum(vec_ca_cb1_norm * cross_nc_norm, axis=1, keepdims=True)
    dot2 = np.sum(vec_ca_cb2_norm * cross_nc_norm, axis=1, keepdims=True)
    dotls = []
    for i,(d1,d2) in enumerate(zip(dot1,dot2)):
        if d1 > 0 and d2 <= 0:
            dotls.append(c1[i])
        elif d2 > 0 and d1 <= 0:
            dotls.append(c2[i])
        else:
            print('both pose fit the L-pose, suppose to choose the close angle to c=o ')
            if ang_cb1[i] > ang_cb2[i]:
                dotls.append(c1[i])
            else:
                dotls.append(c2[i])
    return np.stack(dotls,axis=0)


def calculate_coord(prev_atom1, prev_atom2, prev_atom3, bond_length, angle, dihedral,is_degree = True):


    if is_degree:
        angle = radians(angle)
        dihedral = radians(dihedral)


    v1 = prev_atom2 - prev_atom1
    v2 = prev_atom3 - prev_atom2
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    n = np.cross(v1, v2)
    n /= np.linalg.norm(n)
    nn = np.cross(v1, n)
    nn /= np.linalg.norm(nn)


    x = -bond_length * cos(angle)
    y = bond_length * sin(angle) * cos(dihedral)
    z = bond_length * sin(angle) * sin(dihedral)
    new_vec = x*v1 + y*nn + z*n

    return prev_atom3 + new_vec

def _get_torsion(p0, p1, p2, p3):

    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign( (torch.cross(v1, v2, dim=-1) * v0).sum(-1) )
    dihed = sgn*torch.acos( (n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))
    return dihed
def get_chi_angles(restype, pos14):
    chi_angles = torch.full([4], fill_value=float("inf")).to(pos14)
    base_atom_names = chi_angles_atoms[restype]
    for i, four_atom_names in enumerate(base_atom_names):
        atom_indices = [restype_atom14_name_to_index[restype][a] for a in four_atom_names]
        p = torch.stack([pos14[i] for i in atom_indices])
        torsion = _get_torsion(*torch.unbind(p, dim=0))
        chi_angles[i] = torsion
    return chi_angles
def get_psi_angle(n,ca,c,o):
    return _get_torsion(n,ca,c,o)
def obtain_dihediral_angles(res):
    angle_lis = [0, 0, 0,0]
    for idx, angle in enumerate([res.phi_selection, res.psi_selection, res.omega_selection,res.chi1_selection()]):
        try:
            angle_lis[idx] = angle().dihedral.value()
        except:
            continue
    return angle_lis
def cos_sin_to_degrees(cos_values, sin_values):

    cos = np.asarray(cos_values)
    sin = np.asarray(sin_values)

    if cos.shape != sin.shape:
        raise ValueError("cos_values和sin_values的形状必须相同")

    radians = np.arctan2(sin, cos)

    degrees = np.rad2deg(radians)



    return degrees
def normalize_vector(v):

    return v / np.linalg.norm(v)

def calculate_cg_coordinate(N, CA, CB, dihedral_angle_deg,bond_length):

    v1 = CA - N          
    v2 = CB - CA         

    axis1 = normalize_vector(np.cross(v1, v2))  
    axis2 = normalize_vector(v2)                

    if not np.allclose(axis2, [1, 0, 0]):
        ref_vec = normalize_vector(np.cross(axis2, [1, 0, 0]))
    else:
        ref_vec = normalize_vector(np.cross(axis2, [0, 1, 0]))

    perp_vec = normalize_vector(np.cross(axis2, ref_vec))

    dihedral_rad = math.radians(dihedral_angle_deg)

    rotated_ref = (ref_vec * math.cos(dihedral_rad) + 
                  perp_vec * math.sin(dihedral_rad))



    CG = CB + rotated_ref * bond_length

    return CG
def calculate_cb_coords(n_coord, ca_coord, c_coord,bind_length,o_coord=None
    ):

    N = np.array(n_coord, dtype=np.float64)
    CA = np.array(ca_coord, dtype=np.float64)
    C = np.array(c_coord, dtype=np.float64)
    if o_coord is None:

        O = np.array(o_coord, dtype=np.float64)
    v1 = N - CA  
    v2 = C - CA  

    X = v1 / np.linalg.norm(v1)

    normal = np.cross(v2, v1)
    normal = normal / np.linalg.norm(normal)

    Y = np.cross(normal, X)
    Y = Y / np.linalg.norm(Y)

    Z = np.cross(X, Y)

    ca_cb_bond_length = bind_length

    theta = np.radians(109.5)

    cb_local_x = -np.cos(theta/2)
    cb_local_y = 0.0
    cb_local_z = np.sin(theta/2)

    cb_local = np.array([cb_local_x, cb_local_y, cb_local_z]) * ca_cb_bond_length

    rotation_matrix = np.array([X, Y, Z]).T

    cb_global = CA + np.dot(rotation_matrix, cb_local)

    return cb_global.tolist()
def get_related_atoms(atoms,resname):
    bond = amino_acid_params[resname]['bonds']
    angles = amino_acid_params[resname]['angles']
    relatedbond = []
    relatedangles = []
    for b in bond:
        atomls = b.split('-')
        if atoms in atomls[-1]:
            relatedbond.append(b)
    for b in angles:
        atomls = b.split('-')
        if atoms in atomls[-1]:
            relatedangles.append(b)   
    return  relatedbond,relatedangles
def rotate_point_around_axis(point, axis_point, axis_dir, angle_deg):

    axis_dir = normalize_vector(axis_dir)
    r = Rotation.from_rotvec(math.radians(angle_deg) * axis_dir)
    return r.apply(point - axis_point) + axis_point
def calculate_double_angle_constrained_atom(a1, a2, a3,  
                                           bond_length, 
                                           angle1_deg,   
                                           angle2_deg):  

    v_a1 = normalize_vector(a1 - a2)  
    v_a3 = normalize_vector(a3 - a2)  

    normal_plane = normalize_vector(np.cross(v_a1, v_a3))

    ref_dir = rotate_point_around_axis(v_a3, np.zeros(3), normal_plane, angle2_deg)

    rot_axis = normalize_vector(np.cross(v_a1, ref_dir))

    current_angle = math.degrees(math.acos(np.dot(v_a1, ref_dir)))
    delta_angle = angle1_deg - current_angle

    final_dir = rotate_point_around_axis(ref_dir, np.zeros(3), rot_axis, delta_angle)

    return a2 + final_dir * bond_length


def transform_coordinates(initial_structure, new_positions):



    required_atoms = ['N', 'CA', 'C']
    for atom in required_atoms:
        if atom not in initial_structure or atom not in new_positions:
            raise ValueError(f"初始结构和新位置中都必须包含 {atom} 原子")

    try:
        initial = {
            'N': np.array(initial_structure['N'], dtype=np.float64),
            'CA': np.array(initial_structure['CA'], dtype=np.float64),
            'C': np.array(initial_structure['C'], dtype=np.float64),
        }
    except Exception as e:
        raise ValueError(f"解析初始结构坐标时出错: {str(e)}")

    try:
        new = {
            'N': np.array(new_positions['N'], dtype=np.float64),
            'CA': np.array(new_positions['CA'], dtype=np.float64),
            'C': np.array(new_positions['C'], dtype=np.float64),
        }
    except Exception as e:
        raise ValueError(f"解析新位置坐标时出错: {str(e)}")

    initial_CA_N = initial['N'] - initial['CA']
    initial_CA_C = initial['C'] - initial['CA']


    new_CA_N = new['N'] - new['CA']
    new_CA_C = new['C'] - new['CA']


    for vec, name in [(initial_CA_N, '初始N-CA'), (initial_CA_C, '初始C-CA'), 
                     (new_CA_N, '新N-CA'), (new_CA_C, '新C-CA')]:
        if norm(vec) < 1e-10:
            raise ValueError(f"{name} 向量长度接近零，无法标准化")

    u1 = initial_CA_N / norm(initial_CA_N)
    v1 = initial_CA_C - np.dot(initial_CA_C, u1) * u1
    v1_norm = norm(v1)
    if v1_norm < 1e-10:
        raise ValueError("初始C-CA向量与N-CA向量共线，无法构建坐标系")
    v1 = v1 / v1_norm
    w1 = np.cross(u1, v1)
    w1 = w1 / norm(w1)  

    u2 = new_CA_N / norm(new_CA_N)
    v2 = new_CA_C - np.dot(new_CA_C, u2) * u2
    v2_norm = norm(v2)
    if v2_norm < 1e-10:
        raise ValueError("新C-CA向量与N-CA向量共线，无法构建坐标系")
    v2 = v2 / v2_norm
    w2 = np.cross(u2, v2)
    w2 = w2 / norm(w2)  

    try:
        M1 = np.column_stack((u1, v1, w1))
        M2 = np.column_stack((u2, v2, w2))

        R = M2 @ inv(M1 + 1e-15 * np.eye(3))

        R, _ = np.linalg.qr(R)

        if np.linalg.det(R) < 0:
            R[:, -1] *= -1
    except np.linalg.LinAlgError:
        raise RuntimeError("计算旋转矩阵时出现数值错误，可能是因为初始坐标系退化")

    t = new['CA'] - initial['CA']

    transformed_structure = {}
    for atom, data in initial_structure.items():

        try:
            initial_coord = np.array(data, dtype=np.float64)
            new_coord = R @ (initial_coord - initial['CA']) + new['CA']
            transformed_structure[atom] = new_coord
        except Exception as e:
            raise ValueError(f"计算 {atom} 原子的新坐标时出错: {str(e)}")

    return transformed_structure

def transform_coordinates_ca_cb(initial_structure, new_positions):

    required_atoms = ['CA', 'CB']
    for atom in required_atoms:
        if atom not in initial_structure or atom not in new_positions:
            raise ValueError(f"初始结构和新位置中都必须包含 {atom} 原子")

    try:
        initial = {
            'CA': np.array(initial_structure['CA'], dtype=np.float64),
            'CB': np.array(initial_structure['CB'], dtype=np.float64),
        }
    except Exception as e:
        raise ValueError(f"解析初始结构坐标时出错: {str(e)}")

    try:
        new = {
            'CA': np.array(new_positions['CA'], dtype=np.float64),
            'CB': np.array(new_positions['CB'], dtype=np.float64),
        }
    except Exception as e:
        raise ValueError(f"解析新位置坐标时出错: {str(e)}")

    initial_CA_CB = initial['CB'] - initial['CA']

    if 'N' in initial_structure:
        initial_ref = np.array(initial_structure['N'], dtype=np.float64)
    elif 'C' in initial_structure:
        initial_ref = np.array(initial_structure['C'], dtype=np.float64)
    else:
        raise ValueError("初始结构中需要包含N或C原子作为第三个参考点")

    initial_CA_ref = initial_ref - initial['CA']

    new_CA_CB = new['CB'] - new['CA']

    if 'N' in new_positions:
        new_ref = np.array(new_positions['N'], dtype=np.float64)
    elif 'C' in new_positions:
        new_ref = np.array(new_positions['C'], dtype=np.float64)
    else:
        new_ref = None

    for vec, name in [(initial_CA_CB, '初始CA-CB'), 
                     (new_CA_CB, '新CA-CB')]:
        if norm(vec) < 1e-10:
            raise ValueError(f"{name} 向量长度接近零，无法标准化")

    u1 = initial_CA_CB / norm(initial_CA_CB)
    v1 = initial_CA_ref - np.dot(initial_CA_ref, u1) * u1
    v1_norm = norm(v1)
    if v1_norm < 1e-10:
        raise ValueError("初始参考向量与CA-CB向量共线，无法构建坐标系")
    v1 = v1 / v1_norm
    w1 = np.cross(u1, v1)
    w1 = w1 / norm(w1)  

    u2 = new_CA_CB / norm(new_CA_CB)

    if new_ref is not None:
        new_CA_ref = new_ref - new['CA']
        v2 = new_CA_ref - np.dot(new_CA_ref, u2) * u2
        v2_norm = norm(v2)
        if v2_norm < 1e-10:
            raise ValueError("新参考向量与CA-CB向量共线，无法构建坐标系")
        v2 = v2 / v2_norm
    else:
        v2 = np.cross(w1, u1)
        v2 = np.dot(np.column_stack((u2, np.cross(u2, v2), np.cross(np.cross(u2, v2), u2))).T, v1)
        v2 = v2 / norm(v2)

    w2 = np.cross(u2, v2)
    w2 = w2 / norm(w2)  

    try:
        M1 = np.column_stack((u1, v1, w1))
        M2 = np.column_stack((u2, v2, w2))

        R = M2 @ inv(M1 + 1e-15 * np.eye(3))

        R, _ = qr(R)

        if det(R) < 0:
            R[:, -1] *= -1
    except np.linalg.LinAlgError:
        raise RuntimeError("计算旋转矩阵时出现数值错误，可能是因为初始坐标系退化")

    t = new['CA'] - initial['CA']

    transformed_structure = {}
    for atom, data in initial_structure.items():
        if atom in new_positions:
            transformed_structure[atom] = np.array(new_positions[atom], dtype=np.float64)
            continue

        try:
            initial_coord = np.array(data, dtype=np.float64)
            new_coord = R @ (initial_coord - initial['CA']) + new['CA']
            transformed_structure[atom] = new_coord
        except Exception as e:
            raise ValueError(f"计算 {atom} 原子的新坐标时出错: {str(e)}")

    return transformed_structure

def rotate_around_axis(points, axis_point1, axis_point2, angle):

         axis = axis_point2-axis_point1
         axis = axis / np.linalg.norm(axis)  

         rot_mat = rotation_matrix(angle, axis, axis_point1)[:3, :3]  

         points_rotated = np.dot(points - axis_point1, rot_mat.T) + axis_point1
         return points_rotated

def predict_cbeta_coords(
    n_coord: np.ndarray,    
    ca_coord: np.ndarray,   
    c_coord: np.ndarray,    
    chi1: float,            
) :

    BOND_LENGTH_CA_CB = 1.547  
    BOND_ANGLE_N_CA_CB = np.radians(111.5)  

    for coord in [n_coord, ca_coord, c_coord]:
        if coord.shape != (3,):
            raise ValueError("坐标必须是三维数组 (x, y, z)")

    v_ca_n = n_coord - ca_coord  
    u1 = v_ca_n / np.linalg.norm(v_ca_n)  

    v_ca_c = c_coord - ca_coord  
    v_perp = v_ca_c - np.dot(v_ca_c, u1) * u1  
    u2 = v_perp / np.linalg.norm(v_perp)       

    u3 = np.cross(u1, u2)

    chi1 = chi1

    theta = BOND_ANGLE_N_CA_CB

    dir_local = np.array([
        np.sin(theta) * np.cos(chi1),  
        np.sin(theta) * np.sin(chi1),  
        np.cos(theta)                  
    ])

    transform_matrix = np.column_stack([u1, u2, u3])
    dir_global = transform_matrix @ dir_local  
    cbeta_coord = ca_coord + dir_global * BOND_LENGTH_CA_CB

    return cbeta_coord





def calculate_cb(n_coord, ca_coord, c_coord, o_coord=None):

    N = np.array(n_coord, dtype=np.float64)
    CA = np.array(ca_coord, dtype=np.float64)
    C = np.array(c_coord, dtype=np.float64)

    v_ca_n = N - CA  
    v_ca_c = C - CA  

    x_axis = v_ca_n / np.linalg.norm(v_ca_n)  
    normal = np.cross(v_ca_c, v_ca_n)        
    normal = normal / np.linalg.norm(normal)
    y_axis = np.cross(normal, x_axis)        
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)        

    bond_length = 1.54  
    theta = np.radians(109.5)  

    cb_local1 = np.array([
        -np.cos(theta),
        0,
        np.sin(theta)
    ]) * bond_length

    cb_local2 = np.array([
        -np.cos(theta),
        0,
        -np.sin(theta)
    ]) * bond_length

    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
    cb_candidate1 = CA + np.dot(rotation_matrix, cb_local1)
    cb_candidate2 = CA + np.dot(rotation_matrix, cb_local2)

    if o_coord is not None:
        O = np.array(o_coord)
        v_c_o = O - C
        dot1 = np.dot(np.cross(v_ca_c, v_ca_n), cb_candidate1 - CA)
        dot2 = np.dot(np.cross(v_ca_c, v_ca_n), v_c_o)
        if np.sign(dot1) != np.sign(dot2):
            return cb_candidate1.tolist()
        else:
            return cb_candidate2.tolist()
    else:
        cross_product = np.cross(v_ca_n, v_ca_c)
        dot1 = np.dot(cross_product, cb_candidate1 - CA)
        print(cb_candidate1,cb_candidate2)
        if dot1 > 0:
            return cb_candidate1.tolist()
        else:
            return cb_candidate2.tolist()

from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
import numpy as np
from con_code.common.residue_con import atom_element 
def get_res(atomdic):
    resseq = 1  
    resname = '' 
    segid = "A"  
    residue = Residue(('', resseq, ' '), resname, segid)

    for i, (k,v) in enumerate(atomdic.items(),1):
        atom = Atom(
            name=k,                
            coord=np.array(v, dtype=np.float64),  
            bfactor=0.0,                   
            occupancy=1.0,                 
            altloc=' ',                    
            fullname=f" {k} ",     
            serial_number=i,               
            element=atom_element[k]                
        )
        residue.add(atom)
    return residue
def align_proteins_by_4atoms(initial_structure, new_positions):

    target_residues = []
    required_atoms = ['N', 'CA', 'C','CB']  

    ref_residue = get_res(new_positions)
    ref_atoms = [ref_residue[atom_name] for atom_name in required_atoms]

    superimposer = Superimposer()
    rmsd_values = []
    target_res= get_res(initial_structure)

    target_atoms = [target_res[atom_name] for atom_name in required_atoms]

    superimposer.set_atoms(ref_atoms, target_atoms)


    for atom in target_res:
        atom.transform(superimposer.rotran[0], superimposer.rotran[1])
    new_pos={atoms:selectat(target_res,atoms).coord for atoms in initial_structure}

    return new_pos

def align_points(initial_structure, new_positions):

    A = new_positions['CA']
    B = new_positions['CB'].reshape(-1)
    C = initial_structure['CA']
    D = initial_structure['CB']

    t = A - C  

    D_prime = D + t  

    vec_ad = D_prime - A  
    vec_ab = B - A 
    norm_ad = np.linalg.norm(vec_ad)
    norm_ab = np.linalg.norm(vec_ab)
    if norm_ad == 0 or norm_ab == 0:
        raise ValueError("向量CD或AB为零向量，无法旋转")
    cos_theta = np.dot(vec_ad, vec_ab) / (norm_ad * norm_ab)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  
    theta = np.arccos(cos_theta)
    rotation_axis = np.cross(vec_ad, vec_ab)  
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  
    rotation = R.from_rotvec(theta * rotation_axis)  

    D_prime_centered = D_prime - A  
    D_rotated_centered = rotation.apply(D_prime_centered)  
    B_prime = D_rotated_centered + A  

    new_pos =new_positions.copy()

    new_pos['CB']= B_prime
    for atom in initial_structure:
        if atom not in new_pos.keys():
            pos1 = initial_structure[atom] + t-A

            new_pos[atom] = rotation.apply(pos1)+A



    return new_pos

def align_points_batch(initial_structure, new_pos):

    A = new_pos[:,atom_types.index('CA'),:]
    B = new_pos[:,atom_types.index('CB'),:]
    C = initial_structure[:,atom_types.index('CA'),:]
    D = initial_structure[:,atom_types.index('CB'),:]

    t = B - D  

    C_prime = C + t  
    vec_old = C_prime - B   
    vec_new = A - B         

    norm_old = np.linalg.norm(vec_old, axis=-1, keepdims=True)
    norm_new = np.linalg.norm(vec_new, axis=-1, keepdims=True)

    eps = 1e-8
    vec_old_unit = vec_old / np.clip(norm_old, eps, None)
    vec_new_unit = vec_new / np.clip(norm_new, eps, None)

    cos_theta = np.sum(vec_old_unit * vec_new_unit, axis=-1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)   

    rotation_axis = np.cross(vec_old_unit, vec_new_unit)   
    axis_norm = np.linalg.norm(rotation_axis, axis=-1, keepdims=True)

    rotation_axis_unit = rotation_axis / np.clip(axis_norm, eps, None)
    rotvec = rotation_axis_unit * theta

    parallel_mask = (axis_norm.squeeze(-1) < eps) & (cos_theta.squeeze(-1) > 0.0)
    antiparallel_mask = (axis_norm.squeeze(-1) < eps) & (cos_theta.squeeze(-1) < 0.0)

    rotvec[parallel_mask] = 0.0

    if np.any(antiparallel_mask):
        v = vec_old_unit[antiparallel_mask]  

        x_axis = np.array([1.0, 0.0, 0.0])[None, :]
        alt_axis = np.cross(v, np.repeat(x_axis, len(v), axis=0))
        alt_norm = np.linalg.norm(alt_axis, axis=-1, keepdims=True)

        bad = alt_norm.squeeze(-1) < eps
        if np.any(bad):
            y_axis = np.array([0.0, 1.0, 0.0])[None, :]
            alt_axis[bad] = np.cross(v[bad], np.repeat(y_axis, np.sum(bad), axis=0))
            alt_norm = np.linalg.norm(alt_axis, axis=-1, keepdims=True)

        alt_axis = alt_axis / np.clip(alt_norm, eps, None)
        rotvec[antiparallel_mask] = alt_axis * np.pi

    rotation = R.from_rotvec(rotvec)
    rot_matrices = rotation.as_matrix()
    pos1 = initial_structure + t.reshape(-1,1,3)-B.reshape(-1,1,3)
    rot_matrices = rotation.as_matrix()
    pos1 = np.einsum('nij,nkj->nki', rot_matrices, pos1)

    pos1=pos1+B.reshape(-1,1,3)

    new_pos[:,5:]=pos1[:,5:]


    return new_pos
from Bio.PDB import PDBParser
import numpy as np

def calculate_side_chain_rmsd(
    ref_structure,  
    mob_structure,  
    chain_id: str = None,  
    ignore_hetero: bool = True  
) -> float:

    BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}  

    def get_side_chain_atoms(structure, chain_id: str = None, ignore_hetero: bool = True):

        side_chain_atoms = []
        for model in structure:
            for chain in model:
                if chain_id and chain.id != chain_id:
                    continue
                for residue in chain:
                    if ignore_hetero and residue.id[0] != " ":
                        continue
                    for atom in residue:
                        if atom.id not in BACKBONE_ATOMS:
                            side_chain_atoms.append(atom)
        return side_chain_atoms

    ref_side_atoms = get_side_chain_atoms(ref_structure, chain_id, ignore_hetero)
    mob_side_atoms = get_side_chain_atoms(mob_structure, chain_id, ignore_hetero)

    if len(ref_side_atoms) != len(mob_side_atoms):
        raise ValueError(
            f"参考侧链原子数（{len(ref_side_atoms)}）与目标侧链原子数（{len(mob_side_atoms)}）不匹配！"
            "请确保两个结构的残基顺序、原子类型完全一致（需先对齐序列）。"
        )


    ref_coords = np.array([atom.get_coord() for atom in ref_side_atoms])
    mob_coords = np.array([atom.get_coord() for atom in mob_side_atoms])

    squared_diff = np.sum((ref_coords - mob_coords) ** 2, axis=1)  
    rmsd = np.sqrt(np.mean(squared_diff))

    return rmsd
