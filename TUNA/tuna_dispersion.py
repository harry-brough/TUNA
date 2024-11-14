import numpy as np


def calculate_d2_energy(atoms, bond_length):
 
    """
    
    Requires atoms (list) and bond length (float).

    Calculates the D2 dispersion energy (from Grimme) using the equations in the paper. Uses the C6 and Van der
    Waals radii for H and He provided in the paper, with a s6 factor of 1.2 and damping factor of 20, to match
    the implementation for Hartree-Fock in ORCA.

    Returns the D2 dispersion energy (float).
    
    """

    #These parameters were chosen to match the implementation of Hartree-Fock in ORCA
    s6 = 1.2 
    damping_factor = 20
    
    C6s = []
    vdw_radii = []

   #C6 and VDW radii taken directly from paper 
    atom_properties = {
        "H": {"C6": 2.4284, "vdw_radius": 1.8916},
        "HE": {"C6": 1.3876, "vdw_radius": 1.9124}
    }

    #Looks up atoms in the atom properties dictionary, and builds a list of the C6 values and Van der Waals radii
    for atom in atoms:
        if atom in atom_properties:
            C6s.append(atom_properties[atom]["C6"])
            vdw_radii.append(atom_properties[atom]["vdw_radius"])

    #Makes sure there are two atoms, then calculates the D2 dispersion energy   
    if len(atoms) == 2 and "XH" not in atoms and "XHE" not in atoms:

        C6 = np.sqrt(C6s[0] * C6s[1])
        vdw_sum = vdw_radii[0] + vdw_radii[1]

        f_damp = 1 / (1 + np.exp(-damping_factor * (bond_length / (vdw_sum) - 1)))
        
        #Uses conventional dispersion energy expression, with damping factor to account for short bond lengths
        E_D2 = -s6 * C6 / (bond_length ** 6) * f_damp
        
        return E_D2
        
    else: return 0

