chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  
    [1.0, 1.0, 1.0, 1.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 0.0, 0.0, 0.0],  
    [1.0, 1.0, 1.0, 0.0],  
    [1.0, 1.0, 1.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 1.0, 1.0],  
    [1.0, 1.0, 1.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 0.0, 0.0, 0.0],  
    [1.0, 0.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 1.0, 0.0, 0.0],  
    [1.0, 0.0, 0.0, 0.0],  
]
chi_angles_atoms = {
    'ALA': [],
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
}
atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],

}
resatoms ={'ALA': ['N', 'CA', 'C', 'O', 'CB'],
'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
'GLY': ['N', 'CA', 'C', 'O'],
'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
'TRP': ['N',
'CA',
'C',
'O',
'CB',
'CG',
'CD1',
'CD2',
'NE1',
'CE2',
'CE3',
'CZ2',
'CZ3',
'CH2'],
'TYR': ['N',
'CA',
'C',
'O',
'CB',
'CG',
'CD1',
'CD2',
'CE1',
'CE2',
'CZ',
'OH'],
'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
'UNK': []}
restype_atom14_name_to_index = {
    resname: {name: index for index, name in enumerate(atoms) if name != ""}
    for resname, atoms in restype_name_to_atom14_names.items()
}
amino_acid_params = {
    "Ala": {
        "bonds": {
            "CA-CB": 1.52,  
        },
        "angles": {
            "N-CA-CB": 110.5, 
        }
    },
    "Val": {
        "bonds": {
            "CA-CB": 1.54,
            "CB-CG1": 1.54,
            "CB-CG2": 1.54,
        },
        "angles": {
            "CA-CB-CG1": 109.5,
            "CA-CB-CG2": 109.5,
        }
    },
    "Lys": {
        "bonds": {
            "CA-CB": 1.54,
            "CB-CG": 1.54,
            "CG-CD": 1.54,
            "CD-CE": 1.54,
            "CE-NZ": 1.49,
        },
        "angles": {
            "CA-CB-CG": 109.5,
            "CB-CG-CD": 109.5,
            "CG-CD-CE": 109.5,
            "CD-CE-NZ": 109.5,
        }
    },
    "Gly":{},
    "Ser": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-OG": 1.43,  
        },
        "angles": {
            "CA-CB-OG": 109.5,
            "CB-OG-HO": 108.9  
        }
    },
    "Cys": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-SG": 1.81, 
        },
        "angles": {
            "CA-CB-SG": 114.7,
            "CB-SG-HG": 96.5  
        }
    },
    "Asp": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.51,
            "CG-OD1": 1.25,  
            "CG-OD2": 1.25,
        },
        "angles": {
            "CA-CB-CG": 113.8,
            "CB-CG-OD1": 120.0,  
            "CB-CG-OD2": 120.0
        }
    },
    "Glu": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.53,
            "CG-CD": 1.51,
            "CD-OE1": 1.25,
            "CD-OE2": 1.25
        },
        "angles": {
            "CA-CB-CG": 113.8,
            "CB-CG-CD": 113.8,
            "CG-CD-OE1": 120.0
        }
    },
    "Phe": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.51,
            "CG-CD1": 1.40,  
            "CD1-CE1": 1.40,
            "CE1-CZ": 1.40,
            "CZ-CE2": 1.40,
            "CE2-CD2": 1.40,
            "CD2-CG": 1.40
        },
        "angles": {
            "CA-CB-CG": 113.5,
            "CB-CG-CD1": 120.0,  
            "CG-CD1-CE1": 120.0
        }
    },
    "Tyr": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.51,
            "CG-CD1": 1.40,  
            "CD1-CE1": 1.40,
            "CE1-CZ": 1.40,
            "CZ-CE2": 1.40,
            "CE2-CD2": 1.40,
            "CD2-CG": 1.40,  
            "CZ-OH": 1.36  
        },
        "angles": {
            "CA-CB-CG": 113.5,
            "CB-CG-CD1": 120.0,  
            "CG-CD1-CE1": 120.0,
            "CE2-CZ-OH": 119.0,
            "CZ-OH-HH": 109.5
        }
    },
    "His": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.49,
            "CG-ND1": 1.38,  
            "ND1-CE1": 1.32,
            "CE1-NE2": 1.32,
            "NE2-CD2": 1.38,
            "CD2-CG": 1.35
        },
        "angles": {
            "CA-CB-CG": 113.8,
            "CB-CG-ND1": 126.5,  
            "CG-ND1-CE1": 108.7
        }
    },
    "Trp": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.50,
            "CG-CD1": 1.37,  
            "CD1-NE1": 1.38,
            "NE1-CE2": 1.34,
            "CE2-CZ2": 1.40,
            "CZ2-CH2": 1.40,
            "CH2-CZ3": 1.38,
            "CZ3-CE3": 1.40,
            "CE3-CD2": 1.40
        },
        "angles": {
            "CA-CB-CG": 114.0,
            "CB-CG-CD1": 127.0,  
            "CD1-NE1-CE2": 108.0
        }
    },
        "Leu": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.54,
            "CG-CD1": 1.54,
            "CG-CD2": 1.54
        },
        "angles": {
            "CA-CB-CG": 109.5,
            "CB-CG-CD1": 109.5,
            "CB-CG-CD2": 109.5,
            "CD1-CG-CD2": 109.5 
        }
    },
    "Ile": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG1": 1.54,
            "CB-CG2": 1.54,
            "CG1-CD1": 1.54 
        },
        "angles": {
            "CA-CB-CG1": 109.5,
            "CA-CB-CG2": 109.5,
            "CB-CG1-CD1": 109.5,
            "CG1-CB-CG2": 113.0 
        }
    },
    "Pro": {
        "bonds": {
            "N-CA": 1.46,
            "CA-CB": 1.53,
            "CB-CG": 1.53,
            "CG-CD": 1.53,
            "CD-N": 1.33  
        },
        "angles": {
            "N-CA-CB": 103.8,  
            "CA-CB-CG": 104.0,
            "CB-CG-CD": 105.0,
            "CG-CD-N": 111.0,
            "CD-N-CA": 114.0  
        }
    },
    "Met": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.53,
            "CG-SD": 1.81,
            "SD-CE": 1.80 
        },
        "angles": {
            "CA-CB-CG": 113.0,
            "CB-CG-SD": 113.5,
            "CG-SD-CE": 100.0  
        }
    },
    "Arg": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.53,
            "CG-CD": 1.53,
            "CD-NE": 1.46,
            "NE-CZ": 1.33, 
            "CZ-NH1": 1.33,
            "CZ-NH2": 1.33
        },
        "angles": {
            "CA-CB-CG": 109.5,
            "CB-CG-CD": 109.5,
            "CG-CD-NE": 109.5,
            "CD-NE-CZ": 120.0, 
            "NE-CZ-NH1": 120.0,
            "NE-CZ-NH2": 120.0
        }
    },
    "Gln": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.53,
            "CG-CD": 1.33, 
            "CD-OE1": 1.23, 
            "CD-NE2": 1.33 
        },
        "angles": {
            "CA-CB-CG": 113.8,
            "CB-CG-CD": 111.3,
            "CG-CD-OE1": 120.5,
            "CG-CD-NE2": 117.2
        }
    },
    "Asn": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-CG": 1.52,     
            "CG-OD1": 1.23,     
            "CG-ND2": 1.33   
        },
        "angles": {
            "CA-CB-CG": 113.8,  
            "CB-CG-OD1": 120.5, 
            "CB-CG-ND2": 117.0,  
            "CG-ND2-HD21": 120.0, 
            "CG-ND2-HD22": 120.0
        }
    },
    "Thr": {
        "bonds": {
            "CA-CB": 1.53,
            "CB-OG1": 1.43,      
            "CB-CG2": 1.54     
        },
        "angles": {
            "CA-CB-OG1": 109.5,  
            "CA-CB-CG2": 109.5, 
            "OG1-CB-CG2": 113.0,
            "CB-OG1-HG1": 108.9  
        }
    }

}
METAL = ["LI","NA","K","RB","CS","MG","TL","CU","AG","BE","NI","PT","ZN","CO","PD","AG","CR","FE","V","MN","HG",'GA', 
        "CD","YB","CA","SN","PB","EU","SR","SM","BA","RA","AL","IN","TL","Y","LA","CE","PR","ND","GD","TB","DY","ER",
        "TM","LU","HF","ZR","CE","U","PU","TH"] 
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
three2idx = {k:v for v, k in enumerate(['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
                                        'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
                                        'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                        'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'X'])}
three2self = {v:v for v in ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
                                        'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
                                        'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                        'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP']}
RES_MAX_NATOMS=24
atom_element = {
    "N": "N",       
    "CA": "C",      
    "C": "C",       
    "O": "O",       
    "OXT": "O",     
    
    "CB": "C",      
    "CG": "C",      
    "CG1": "C",     
    "CG2": "C",     
    "CD": "C",      
    "CD1": "C",     
    "CD2": "C",     
    "CE": "C",      
    "CE1": "C",     
    "CE2": "C",     
    "CE3": "C",     
    "CZ": "C",      
    "CZ2": "C",     
    "CZ3": "C",     
    "CH2": "C",     
    
    "OG": "O",      
    "OG1": "O",     
    "OH": "O",      
    "OD1": "O",     
    "OD2": "O",     
    "OE1": "O",     
    "OE2": "O",     
    
    "SG": "S",      
    "SD": "S",      
    
    "ND1": "N",     
    "ND2": "N",     
    "NE": "N",      
    "NE1": "N",     
    "NE2": "N",     
    "NZ": "N",      
    "NH1": "N",     
    "NH2": "N"      
}