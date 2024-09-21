import numpy as np
from termcolor import colored
import sys
import tuna_util as util


def calculate_centre_of_mass(masses, coordinates): return np.einsum("i,ij->", masses, coordinates) / np.sum(masses)

def calculate_electronic_dipole_moment(P, D): return -np.einsum("ij,ij->",P,D)

def calculate_reduced_mass(masses): return np.prod(masses) / np.sum(masses)

 
def calculate_nuclear_dipole_moment(centre_of_mass, Z_list, coordinates): 

    nuclear_dipole_moment = 0
    
    for i in range(len(Z_list)): nuclear_dipole_moment += (coordinates[i][2] - centre_of_mass) * Z_list[i]
        
    return nuclear_dipole_moment
   

def calculate_rotational_constant(masses, coordinates):

    bond_length = np.linalg.norm(coordinates[1] - coordinates[0])
    reduced_mass = calculate_reduced_mass(masses)
    
    rotational_constant_hartree = 1 / (2 * reduced_mass * bond_length ** 2)  
    rotational_constant_per_bohr = rotational_constant_hartree / (util.constants.h * util.constants.c)
    
    rotational_constant_per_cm = rotational_constant_per_bohr / (100 * util.constants.bohr_in_metres)
    rotational_constant_GHz = util.constants.per_cm_in_GHz * rotational_constant_per_cm
    
    return rotational_constant_per_cm, rotational_constant_GHz


def calculate_koopman_parameters(epsilons, n_occ):


    ionisation_energy = -1 * epsilons[n_occ - 1]
        
    if len(epsilons) > n_occ: 
    
        electron_affinity = -1 * epsilons[n_occ]
        homo_lumo_gap = ionisation_energy - electron_affinity
        
    else: 
    
        electron_affinity = "---"
        homo_lumo_gap = "---"
        
        print(colored("WARNING: Size of basis is too small for electron affinity calculation!","light_yellow"))

    return ionisation_energy, electron_affinity, homo_lumo_gap
 
 
 
def construct_electron_density(P, grid_density, molecule):

    print("\n Beginning electron density surface plot calculation...\n")

    print(" Setting up grid...   ", end="")
    
    coordinates = [molecule.coordinates[0][2], molecule.coordinates[1][2]]
    start = coordinates[0] - 4
    
    x = np.arange(start, coordinates[0] + 4 + grid_density, grid_density)
    y = np.arange(start, coordinates[0] + 4 + grid_density, grid_density)
    z = np.arange(start, coordinates[1] + 4 + grid_density, grid_density)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    
    print("[Done]")
    
    print(" Generating electron density cube...   ", end=""); sys.stdout.flush()
    
    n = 0

    atomic_orbitals = []

    for orbital in molecule.atomic_orbitals:

        a = 0

        for pg in orbital:  
        
            a += pg.N * pg.coeff * np.exp(-pg.alpha * ((X - pg.coordinates[0])**2 + (Y - pg.coordinates[1])**2 + (Z - pg.coordinates[2])**2))
        
        
        atomic_orbitals.append(a)
    
    atomic_orbitals = np.array(atomic_orbitals)
    
    n = np.einsum("mn,mijk,nijk->ijk", P, atomic_orbitals, atomic_orbitals)
    
    normalisation = np.trapz(np.trapz(np.trapz(n,z),y), x)
    n *= molecule.n_doubly_occ * 2 / normalisation

    print("[Done]")
    print(" Generating surface plot...   ", end="")
    sys.stdout.flush()
    isovalue = 0.06
    
    from skimage import measure
    import plotly.graph_objects as go
    
    verts, faces, _, _ = measure.marching_cubes(n, isovalue, spacing=(grid_density, grid_density, grid_density))
    intensity = np.full(len(verts), isovalue)
    
    fig = go.Figure(data=[go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],intensity=intensity,colorscale='Agsunset',opacity=0.5)])
    fig.update_layout(scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),bgcolor='rgb(255, 255, 255)'),margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(scene_camera=dict(eye=dict(x=0.5, y=2.5, z=0.5)))
    print("[Done]\n")
    
    fig.show()
    
    return n


def print_energy_components(nuclear_electron_energy, kinetic_energy, exchange_energy, coulomb_energy, V_NN, calculation):


    one_electron_energy = nuclear_electron_energy + kinetic_energy
    two_electron_energy = exchange_energy + coulomb_energy
    electronic_energy = one_electron_energy + two_electron_energy
    total_energy = electronic_energy + V_NN
            
    util.log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation)      
    util.log("              Energy Components       ", calculation)
    util.log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation)
            

    util.log(f"  Kinetic energy:              {kinetic_energy:.10f}", calculation)

    util.log(f"  Coulomb energy:              {coulomb_energy:.10f}", calculation)
    util.log(f"  Exchange energy:            {exchange_energy:.10f}", calculation)
    util.log(f"  Nuclear repulsion energy:    {V_NN:.10f}", calculation)
    util.log(f"  Nuclear attraction energy:  {nuclear_electron_energy:.10f}\n", calculation)      

    util.log(f"  One-electron energy:        {one_electron_energy:.10f}", calculation)
    util.log(f"  Two-electron energy:         {two_electron_energy:.10f}", calculation)
    util.log(f"  Electronic energy:          {electronic_energy:.10f}\n", calculation)
            
    util.log(f"  Total energy:               {total_energy:.10f}", calculation)
    util.log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation)  

    return



def calculate_population_analysis(P, S, R, ao_ranges, atoms, Z_list):

    """
    
    Requires density matrix, overlap matrix, spin density matrix (alpha minus beta density matrices), list of atoms and list of charges.
    
    Performs Mulliken, Lowdin and Mayer population analysis. Does all of these together to share for loop infrastructure.
    
    Returns the Mulliken bond order, charges and total charge, the Lowin bond order, charges and total charge, and the Mayer bond order, free and total valences.
    
    """

    PS = P @ S
    RS = R @ S

    #Diagonalises overlap matrix to form density matrix in orthogonalised Lowdin basis
    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    P_lowdin = S_sqrt @ P @ S_sqrt

    #Initialisation of various variables
    bond_order_mayer = 0
    bond_order_lowdin = 0
    bond_order_mulliken = 0

    total_valences = [0, 0]

    populations_mulliken = [0, 0]
    populations_lowdin = [0, 0]
    charges_mulliken = [0, 0]
    charges_lowdin = [0, 0]


    #Sums over the ranges of each atomic orbital over atom A, then atom B to build the three bond orders
    for i in range(ao_ranges[0]):
        for j in range(ao_ranges[0], ao_ranges[0] + ao_ranges[1]):

            bond_order_mayer += PS[i,j] * PS[j,i] + RS[i,j] * RS[j,i]
            bond_order_lowdin += P_lowdin[i,j] ** 2
            bond_order_mulliken += 2 * P[i,j] * S[i,j]
    
    #Sums over atoms, then corresponding ranges of atomic orbitals in the density matrix, to build the valences and populations
    for atom in range(len(atoms)):

        if atom == 0: atomic_ranges = list(range(ao_ranges[0]))
        elif atom == 1: atomic_ranges = list(range(ao_ranges[0], ao_ranges[0] + ao_ranges[1]))

        for i in atomic_ranges:
            
            populations_lowdin[atom] += P_lowdin[i,i] 
            populations_mulliken[atom] += PS[i,i]

            for j in atomic_ranges:

                total_valences[atom] += PS[i,j] * PS[j,i]

        charges_mulliken[atom] = Z_list[atom] - populations_mulliken[atom]
        charges_lowdin[atom] = Z_list[atom] - populations_lowdin[atom]

        total_valences[atom] = 2 * populations_mulliken[atom] - total_valences[atom]


    total_charges_mulliken = np.sum(charges_mulliken)
    total_charges_lowdin = np.sum(charges_lowdin)

    free_valences = np.array(total_valences) - bond_order_mayer


    return bond_order_mulliken, charges_mulliken, total_charges_mulliken, bond_order_lowdin, charges_lowdin, total_charges_lowdin, bond_order_mayer, free_valences, total_valences




def format_population_analysis_output(charges_mulliken, charges_lowdin, total_charges_mulliken, bond_order_mulliken, bond_order_lowdin, free_valences, total_valences, atoms):
    
    """
    
    Requires Mulliken and Lowdin charges, total Mulliken charges, Mulliken and Lowdin bond orders, free and total Mayer valences and atoms list.

    Formats these values appropriately for the terminal output so the decimal points are aligned and negative signs don't mess things up.

    Returns formatted sizes of blank spaces (1, 2 and 3), as well as these formatted values.


    """

    space = "" if total_charges_mulliken < 0 else "  "
    space2 = "" if bond_order_mulliken < 0 else "  "
    space3 = "" if bond_order_lowdin < 0 else " "

    atoms_formatted = []
    free_valences_formatted = []

    #Combined into one for loop for performance
    for i, atom in enumerate(atoms):
    
        atom = atom.lower().capitalize()
        atom = atom + "  :" if len(atom) == 1 else atom + " :"
        atoms_formatted.append(atom)

        if free_valences[i] > 0: free_valences_formatted.append(f" {free_valences[i]:.5f}")
        else: free_valences_formatted.append(f"{free_valences[i]:.5f}")

        if total_valences[i] > 0: total_valences[i] = f" {total_valences[i]:.5f}"
        else: total_valences[i] = f"{total_valences[i]:.5f}"

        if charges_mulliken[i] > 0: charges_mulliken[i] = f" {charges_mulliken[i]:.5f}"
        else: charges_mulliken[i] = f"{charges_mulliken[i]:.5f}"

        if charges_lowdin[i] > 0: charges_lowdin[i] = f" {charges_lowdin[i]:.5f}"
        else: charges_lowdin[i] = f"{charges_lowdin[i]:.5f}"

    
    return space, space2, space3, charges_mulliken, charges_lowdin, free_valences_formatted, total_valences, atoms_formatted 



def post_scf_output(molecule, calculation, epsilons, molecular_orbitals, P, S, ao_ranges, D, P_alpha, P_beta):

    print("\n Beginning calculation of TUNA properties... ")
        
    method = calculation.method
    additional_print = calculation.additional_print
    n_electrons = molecule.n_electrons
    n_doubly_occ = molecule.n_doubly_occ
    masses = molecule.masses
    coordinates = molecule.coordinates
    atoms = molecule.atoms
    Z_list = molecule.Z_list
    molecular_structure = molecule.molecular_structure

    if method == "MP2" and calculation.reference == "RHF": print("\n Using the MP2 unrelaxed density for property calculations.")
    elif method == "SCS-MP2": util.warning("The SCS-MP2 density is not implemented! Using unscaled MP2 density for property calculations.")
    elif method == "UMP2" or method == "MP2" and calculation.reference == "UHF" or method == "MP3" and calculation.reference == "UHF" or method == "UMP3": util.warning("Using the unrestricted Hartree-Fock density for property calculations.")
    elif method == "MP3" or method == "SCS-MP3": util.warning("Using the Hartree-Fock density for property calculations.")

    if additional_print:
            
        print("\n Molecular orbital eigenvalues:\n")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~")
        print("  N     Occ    Epsilon ")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~")
            
        if n_electrons > 1: occupancies = [2] * n_doubly_occ + [0] * int((len(epsilons) - n_doubly_occ))
        else: occupancies = [1] + [0] * (len(epsilons) - 1)
            
        for i in range(len(epsilons)):
            
            if i < 9: print(f"  {i + 1}     {occupancies[i]}     {np.round(epsilons[i],decimals=6)}")
            else: print(f" {i + 1}     {occupancies[i]}     {np.round(epsilons[i],decimals=6)}")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~\n")

        print(" Molecular orbital coefficients:\n")

        symbol_list = []
        n_list = []
        switch_value = 0

        for mo in range(len(epsilons)):
            
            if n_electrons / 2 > mo: occ = "(Occupied)"
            else: occ = "(Virtual)"

            print(f"\n MO {mo+1} {occ}\n")
                
            for i, atom in enumerate(molecule.mol):
                for j, orbital in enumerate(atom):
                    
                    symbol_list.append(atoms[i])                  
                    n_list.append(j + 1)
                    
                    if i == 1 and j == 0 and mo == 0: switch_value = len(symbol_list) - 1
                
                
            for k in range(len(molecular_orbitals.T[mo])):
                if k == switch_value: print("")
                print("  " + symbol_list[k].lower().capitalize() + f"  {n_list[k]}s  :  " + str(np.round(molecular_orbitals.T[mo][k], decimals=4)))
                
        print("")

    if molecule.n_occ % 2 == 0: ionisation_energy, electron_affinity, homo_lumo_gap = calculate_koopman_parameters(epsilons, molecule.n_doubly_occ)
    else: ionisation_energy, electron_affinity, homo_lumo_gap = calculate_koopman_parameters(epsilons, molecule.n_occ)


    if type(electron_affinity) == np.float64: electron_affinity = np.round(electron_affinity,decimals=6)
    if type(homo_lumo_gap) == np.float64: homo_lumo_gap = np.round(homo_lumo_gap,decimals=6)
        
    print(f"\n Koopmans' theorem ionisation energy: {ionisation_energy:.6f}")
    print(f" Koopmans' theorem electron affinity: {electron_affinity}")
    print(f" Energy gap between HOMO and LUMO: {homo_lumo_gap}")


    if len(molecule.atoms) != 1 and "XH" not in atoms and "XHE" not in atoms:

        B_per_cm, B_GHz = calculate_rotational_constant(masses, coordinates)
                
        print(f"\n Rotational constant (GHz): {B_GHz:.3f}")

        centre_of_mass = calculate_centre_of_mass(masses, coordinates)

        print(f"\n Dipole moment origin is the centre of mass, {util.bohr_to_angstrom(centre_of_mass):.4f} angstroms from the first atom.")

        D_nuclear = calculate_nuclear_dipole_moment(centre_of_mass, Z_list, coordinates)        
        D_electronic = calculate_electronic_dipole_moment(P, D)


        total_dipole = D_nuclear + D_electronic

        print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("                Dipole Moment")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        print(f"  Nuclear: {D_nuclear:.6f}    Electronic: {D_electronic:.6f}\n")
        print(f"  Total: {total_dipole:.6f}",end="")
            
        if total_dipole > 0.00001:

            print("        " + molecular_structure, end="")
            print("  +--->")

        elif total_dipole < -0.00001:

            print("        " + molecular_structure, end="")
            print("  <---+")

        else: print(f"           {molecular_structure}")

        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        
        bond_order_mulliken, charges_mulliken, total_charges_mulliken, bond_order_lowdin, charges_lowdin, total_charges_lowdin, bond_order_mayer, free_valences, total_valences = calculate_population_analysis(P, S, P_alpha - P_beta, ao_ranges, atoms, Z_list)
    
        space, space2, space3, charges_mulliken, charges_lowdin, free_valences, total_valences, atoms_formatted = format_population_analysis_output(charges_mulliken, charges_lowdin, total_charges_mulliken, bond_order_mulliken, bond_order_lowdin, free_valences, total_valences, atoms)


        print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("      Mulliken Charges                Lowdin Charges              Mayer Free, Bonded, Total Valence")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"  {atoms_formatted[0]}  {charges_mulliken[0]}                 {atoms_formatted[0]}  {charges_lowdin[0]}                 {atoms_formatted[0]}   {free_valences[0]},  {bond_order_mayer:.5f}, {total_valences[0]}")
        print(f"  {atoms_formatted[1]}  {charges_mulliken[1]}                 {atoms_formatted[1]}  {charges_lowdin[1]}                 {atoms_formatted[1]}   {free_valences[1]},  {bond_order_mayer:.5f}, {total_valences[1]}")
        print(f"\n  Sum of charges: {total_charges_mulliken:.5f}   {space}   Sum of charges: {total_charges_lowdin:.5f}") 
        print(f"  Bond order: {bond_order_mulliken:.5f}      {space2}    Bond order: {bond_order_lowdin:.5f}      {space3}     Bond order: {bond_order_mayer:.5f}") 
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    return