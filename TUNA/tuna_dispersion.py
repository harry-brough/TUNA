import numpy as np


def calculate_d2_energy(atoms, bond_length):
 
    s6 = 1.2 
    damping_factor = 20
    
    C6s = []
    vdw_radii = []
    
    atom_properties = {
        "H": {"C6": 2.4284, "vdw_radius": 1.8916},
        "HE": {"C6": 1.3876, "vdw_radius": 1.9124}
    }

    for atom in atoms:
        if atom in atom_properties:
            C6s.append(atom_properties[atom]["C6"])
            vdw_radii.append(atom_properties[atom]["vdw_radius"])
       
    if len(atoms) == 2 and "XH" not in atoms and "XHE" not in atoms:
        C6 = np.sqrt(C6s[0] * C6s[1])
        vdw_sum = vdw_radii[0] + vdw_radii[1]

        f_damp = 1 / (1 + np.exp(-damping_factor * (bond_length / (vdw_sum) - 1)))

        E_D2 = -s6 * C6 / (bond_length ** 6) * f_damp
        
        return E_D2
        
    else: return 0

