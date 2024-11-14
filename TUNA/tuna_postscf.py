import numpy as np
from tuna_util import *

def calculate_electronic_dipole_moment(P, D): 
    
    """
    
    Requires density matrix (array) and dipole integrals (array).

    Calculates and returns electronic dipole moment (float) by efficient tensor contraction.
    
    """
    
    return -np.einsum("ij,ij->",P,D,optimize=True)



def calculate_reduced_mass(masses): 
    
    """
    
    Requires masses (array).

    Calculates and returns the reduced mass (float).

    """
    
    return np.prod(masses) / np.sum(masses)

 

def calculate_nuclear_dipole_moment(centre_of_mass, charges, coordinates): 

    """
    
    Requires centre of mass (float), nuclear charges (list), coordinates (array).

    Calculates nuclear dipole moment by iterating through nuclei.
    
    Returns nuclear dipole moment (float).
    
    """

    nuclear_dipole_moment = 0
    
    for i in range(len(charges)): 
        
        nuclear_dipole_moment += (coordinates[i][2] - centre_of_mass) * charges[i]
        
    return nuclear_dipole_moment
   


def calculate_rotational_constant(masses, coordinates):

    """
    
    Requires masses (list) and coordinates (array).

    Calculates the rotational constant for a linear molecule and convertst to various units.
    
    Returns the rotational constant in per cm and GHz.

    """

    bond_length = np.linalg.norm(coordinates[1] - coordinates[0])
    reduced_mass = calculate_reduced_mass(masses)
    
    #Standard equation for linear molecule's rotational constant
    rotational_constant_hartree = 1 / (2 * reduced_mass * bond_length ** 2)

    #Various unit conversions  
    rotational_constant_per_bohr = rotational_constant_hartree / (constants.h * constants.c)
    rotational_constant_per_cm = rotational_constant_per_bohr / (100 * constants.bohr_in_metres)
    rotational_constant_GHz = constants.per_cm_in_GHz * rotational_constant_per_cm
    
    return rotational_constant_per_cm, rotational_constant_GHz


def calculate_koopman_parameters(epsilons, n_occ):

    """
    
    Requires Hartree-Fock eigenvalues (array) and number of occupied orbitals (int).

    Calculates electron affinity and ionisation energy using Koopman's theorem with eigenvalues, as well as the difference in orbital energy
    of the HOMO and LUMO.

    Returns ionisation energy (float), electron affinity (float) and HOMO-LUMO gap (float).

    """

    #IE = -HOMO
    ionisation_energy = -1 * epsilons[n_occ - 1]

    #As long as LUMO exists, EA = -LUMO    
    if len(epsilons) > n_occ: 
    
        electron_affinity = -1 * epsilons[n_occ]
        homo_lumo_gap = ionisation_energy - electron_affinity
        
    else: 
    
        electron_affinity = "---"
        homo_lumo_gap = "---"
        
        warning("WARNING: Size of basis is too small for electron affinity calculation!")


    return ionisation_energy, electron_affinity, homo_lumo_gap
 
 


def print_energy_components(nuclear_electron_energy, kinetic_energy, exchange_energy, coulomb_energy, V_NN, calculation):

    """
    
    Requires nuclear-electron energy (float), kinetic energy (float), exchange energy (float), Coulomb energy (float), 
    nuclear-nuclear repulsion energy (float) and calculation (Calculation).

    Calculates the one- and two-electron contributions to energy, and prints all the energy components out to the console.
    
    """

    #Adds up different energy components
    one_electron_energy = nuclear_electron_energy + kinetic_energy
    two_electron_energy = exchange_energy + coulomb_energy
    electronic_energy = one_electron_energy + two_electron_energy
    total_energy = electronic_energy + V_NN
            
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)      
    log("              Energy Components       ", calculation, 2)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
            

    log(f"  Kinetic energy:              {kinetic_energy:.10f}", calculation, 2)

    log(f"  Coulomb energy:              {coulomb_energy:.10f}", calculation, 2)
    log(f"  Exchange energy:            {exchange_energy:.10f}", calculation, 2)
    log(f"  Nuclear repulsion energy:    {V_NN:.10f}", calculation, 2)
    log(f"  Nuclear attraction energy:  {nuclear_electron_energy:.10f}\n", calculation, 2)      

    log(f"  One-electron energy:        {one_electron_energy:.10f}", calculation, 2)
    log(f"  Two-electron energy:         {two_electron_energy:.10f}", calculation, 2)
    log(f"  Electronic energy:          {electronic_energy:.10f}\n", calculation, 2)
            
    log(f"  Total energy:               {total_energy:.10f}", calculation, 2)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)  



def calculate_population_analysis(P, S, R, ao_ranges, atoms, charges):

    """
    
    Requires density matrix (array), overlap matrix (array), spin density matrix (alpha minus beta density matrices, array), list of atoms (list)
    and list of charges (list).
    
    Performs Mulliken, Lowdin and Mayer population analysis. Does all of these together to share for loop infrastructure.
    
    Returns the Mulliken bond order (float), charges and total charge (floats), the Lowin bond order (float), charges and total charge (floats), and the Mayer bond order (float),
    free and total valences (floats).
    
    """

    PS = P @ S
    RS = R @ S

    #Diagonalises overlap matrix to form density matrix in orthogonalised Lowdin basis
    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    P_lowdin = S_sqrt @ P @ S_sqrt

    #Initialisation of various variables
    bond_order_mayer = bond_order_lowdin = bond_order_mulliken = 0


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
            total_valences[atom] += np.einsum("j,j->", PS[i, atomic_ranges], PS[atomic_ranges, i],optimize=True)

        charges_mulliken[atom] = charges[atom] - populations_mulliken[atom]
        charges_lowdin[atom] = charges[atom] - populations_lowdin[atom]

        total_valences[atom] = 2 * populations_mulliken[atom] - total_valences[atom]

    #Adds up total charges and calculates free valences from total and bonded valences
    total_charges_mulliken = np.sum(charges_mulliken)
    total_charges_lowdin = np.sum(charges_lowdin)

    free_valences = np.array(total_valences) - bond_order_mayer


    return bond_order_mulliken, charges_mulliken, total_charges_mulliken, bond_order_lowdin, charges_lowdin, total_charges_lowdin, bond_order_mayer, free_valences, total_valences




def format_population_analysis_output(charges_mulliken, charges_lowdin, total_charges_mulliken, bond_order_mulliken, bond_order_lowdin, free_valences, total_valences, atoms):
    
    """
    
    Requires Mulliken and Lowdin charges (array), total Mulliken charges (float), Mulliken and Lowdin bond orders (floats), free and total Mayer valences (arrays) and atoms (list).

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



def print_molecular_orbital_eigenvalues(calculation, reference, n_doubly_occ, n_alpha, n_beta, epsilons, epsilons_alpha, epsilons_beta):

    """

    Requires calculation (Calculation), reference (string), number of doubly occupied orbitals (int), number of alpha and beta electrons (ints), Hartree-Fock
    eigenvalues (array), and separate alpha and beta eigenvalues (arrays).

    Prints out nicely formatted and organised eigenvalues for all orbitals (for RHF) and alpha and beta orbitals separately (for UHF).

    """


    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n     Molecular Orbital Eigenvalues\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 3)

    def print_eigenvalue_header(calculation):

        log("  ~~~~~~~~~~~~~~~~~~~~~~~", calculation, 3)
        log("   N    Occ     Epsilon ", calculation, 3)
        log("  ~~~~~~~~~~~~~~~~~~~~~~~", calculation, 3)


    #Prints all the eigenvalues for an array of epsilons and corresponding occupancies, used by UHF and RHF 
    def print_eigenvalues(epsilons, occupancies):

        for i, epsilon in enumerate(epsilons):
            
            space = "" if epsilon < 0 else " "

            if i < 9: log(f"   {i + 1}     {occupancies[i]}    {space} {np.round(epsilon,decimals=6)}", calculation, 3)
            else: log(f"  {i + 1}     {occupancies[i]}  {space}   {np.round(epsilon,decimals=6)}", calculation, 3)

        log("  ~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, 3)


    #Prints alpha and beta eigenvalues separately
    if reference == "UHF":

        log("\n  Alpha orbital eigenvalues:\n", calculation, 3)
        print_eigenvalue_header(calculation)
        
        #Occupied orbitals are alpha electrons only
        occupancies = [1] * n_alpha + [0] * int((len(epsilons_alpha) - n_alpha))

        print_eigenvalues(epsilons_alpha, occupancies)

        #If beta electrons are present, the orbitals are occupied only upto the number of beta electrons
        if epsilons_beta is not None:

            log("  Beta orbital eigenvalues:\n", calculation, 3)
            print_eigenvalue_header(calculation)
            
            occupancies = [1] * n_beta + [0] * int((len(epsilons_beta) - n_beta))

            print_eigenvalues(epsilons_beta, occupancies)


    elif reference == "RHF":

        print_eigenvalue_header(calculation)
        
        #Occupied orbitals (doubly occupied) depend on number electron pairs
        occupancies = [2] * n_doubly_occ + [0] * int((len(epsilons) - n_doubly_occ))

        print_eigenvalues(epsilons, occupancies)





def print_molecular_orbital_coefficients(molecule, atoms, calculation, reference, epsilons, epsilons_alpha, epsilons_beta, n_alpha, n_beta, n_doubly_occ, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta):
    
    """
    
    Requires molecule (Molecule), atoms (list), calculation (Calculation), reference (string), epsilons (array), alpha and beta epsilons (arrays), number of 
    alpha and beta electrons (ints), number of double occupied orbitals (int), moelcular orbitals, alpha and beta molecular orbitals (arrays).

    Prints the coefficients of all the molecular orbitals (or separated alpha and beta orbitals for UHF), in a nice and organised way.
    
    """

    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n     Molecular Orbital Coefficients\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 3)

    #Initialises various quantities
    symbol_list = []
    n_list = []
    switch_value = 0

    #Builds a list of atomic symbols and number of atomic orbitals per atom
    for i, atom in enumerate(molecule.mol):
        for j, _ in enumerate(atom):
            
            symbol_list.append(atoms[i])                  
            n_list.append(j + 1)
            
            #Determines the index at which the orbitals switch over from one atomic centre to the other
            if i == 1 and j == 0: 
                
                switch_value = len(symbol_list) - 1

    #Prints out coefficients for each orbital, as well as if each is occupied or virtual
    def print_coeffs(switch_value, calculation, symbol_list, molecular_orbitals, n_list, eps, n):

        for mo in range(len(eps)):
            
            if n > mo: occ = "(Occupied)"
            else: occ = "(Virtual)"

            log(f"\n   MO {mo+1} {occ}\n", calculation, 3)
                
                
            for k in range(len(molecular_orbitals.T[mo])):
                
                #Formats ghost atoms nicely
                if "X" in symbol_list[k]: 

                    symbol_list[k] = symbol_list[k].split("X")[1]
                    symbol_list[k] = "X" + symbol_list[k].lower().capitalize()
                    
                else: symbol_list[k] = symbol_list[k].lower().capitalize()

                if k == switch_value and len(atoms) == 2: log("", calculation, 3)

                log("    " + symbol_list[k] + f"  {n_list[k]}s  :  " + str(np.round(molecular_orbitals.T[mo][k], decimals=4)), calculation, 3)


    #For UHF calculations, do all of the above but separately for alpha and beta orbitals
    if reference == "UHF":

        log("\n  Alpha orbital coefficients:", calculation, 3)

        print_coeffs(switch_value, calculation, symbol_list, molecular_orbitals_alpha, n_list, epsilons_alpha, n_alpha)

        if epsilons_beta is not None:

            log("\n\n  Beta orbital coefficients:", calculation, 3)

            print_coeffs(switch_value, calculation, symbol_list, molecular_orbitals_beta, n_list, epsilons_beta, n_beta)

    #For RHF calculations, do all of the above for the combined doubly occupied orbitals
    else:
        
        print_coeffs(switch_value, calculation, symbol_list, molecular_orbitals, n_list, epsilons, n_doubly_occ)


    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 3)





def post_scf_output(molecule, calculation, epsilons, molecular_orbitals, P, S, ao_ranges, D, P_alpha, P_beta, epsilons_alpha, epsilons_beta, molecular_orbitals_alpha, molecular_orbitals_beta):

    """
    
    Requires molecule (Molecule), calculation (Calculation), epsilons (array), molecular orbitals (array), density matrix (array), overlap matrix (array), ranges of atomic
    orbitals (array), dipole integrals (array), alpha and beta density matrices (arrays), alpha and beta epsilons (arrays), alpha and beta molecular orbitals (arrays).

    Prints out molecular orbitals, eigenvalues, Koopman's theorem parameters, dipole moment information, population analysis and rotational constant information.
    
    """

    log("\n Beginning calculation of TUNA properties... ", calculation, 3)

    #Unpacks useful calculation quantities    
    method = calculation.method

    #Unpacks useful molecular quantities
    n_doubly_occ = molecule.n_doubly_occ
    n_alpha = molecule.n_alpha
    n_beta = molecule.n_beta
    reference = calculation.reference
    masses = molecule.masses
    coordinates = molecule.coordinates
    atoms = molecule.atoms
    charges = molecule.charges
    molecular_structure = molecule.molecular_structure

    #Specifies which density matrix is used for the property calculations
    if method == "MP2" and calculation.reference == "RHF": log("\n Using the MP2 unrelaxed density for property calculations.", calculation, 1)
    elif method == "SCS-MP2": warning("The SCS-MP2 density is not implemented! Using unscaled MP2 density for property calculations.")
    elif method == "UMP2" or method == "MP2" and calculation.reference == "UHF" or method == "MP3" and calculation.reference == "UHF" or method == "UMP3": warning("Using the unrestricted Hartree-Fock density for property calculations.")
    elif method == "MP3" or method == "SCS-MP3": warning("Using the Hartree-Fock density for property calculations.")

    #Prints molecular orbital eigenvalues and coefficients
    print_molecular_orbital_eigenvalues(calculation, reference, n_doubly_occ, n_alpha, n_beta, epsilons, epsilons_alpha, epsilons_beta)
    print_molecular_orbital_coefficients(molecule, atoms, calculation, reference, epsilons, epsilons_alpha, epsilons_beta, n_alpha, n_beta, n_doubly_occ, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta)

    #Prints Koopman theorem parameters
    if calculation.reference == "RHF":
        
        ionisation_energy, electron_affinity, homo_lumo_gap = calculate_koopman_parameters(epsilons, molecule.n_doubly_occ)

        if type(electron_affinity) == np.float64: electron_affinity = np.round(electron_affinity,decimals=6)
        if type(homo_lumo_gap) == np.float64: homo_lumo_gap = np.round(homo_lumo_gap,decimals=6)
            
        log(f"\n Koopmans' theorem ionisation energy: {ionisation_energy:.6f}", calculation, 2)
        log(f" Koopmans' theorem electron affinity: {electron_affinity}", calculation, 2)
        log(f" Energy gap between HOMO and LUMO: {homo_lumo_gap}", calculation, 2)

    #As long as there are two real atoms present, calculates rotational constant and dipole moment information
    if len(molecule.atoms) != 1 and not any("X" in atom for atom in atoms):

        _, B_GHz = calculate_rotational_constant(masses, coordinates)
                
        log(f"\n Rotational constant (GHz): {B_GHz:.3f}", calculation, 2)

        #Calculates centre of mass for dipole moment calculations
        centre_of_mass = calculate_centre_of_mass(masses, coordinates)

        log(f"\n Dipole moment origin is the centre of mass, {bohr_to_angstrom(centre_of_mass):.4f} angstroms from the first atom.", calculation, 2)

        D_nuclear = calculate_nuclear_dipole_moment(centre_of_mass, charges, coordinates)        
        D_electronic = calculate_electronic_dipole_moment(P, D)
        total_dipole = D_nuclear + D_electronic

        log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
        log("                Dipole Moment", calculation, 2)
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)

        log(f"  Nuclear: {D_nuclear:.6f}    Electronic: {D_electronic:.6f}\n", calculation, 2)
        log(f"  Total: {total_dipole:.6f}", calculation, 2, end="")

        #Prints direction of dipole moment, where plus indicates positive side    
        if total_dipole > 0.00001:

            log("        " + molecular_structure, calculation, 2, end="")
            log("  +--->", calculation, 2)

        elif total_dipole < -0.00001:

            log("        " + molecular_structure, calculation, 2, end="")
            log("  <---+", calculation, 2)

        #If there's no dipole moment, just print out the molecular structure
        else: log(f"           {molecular_structure}", calculation, 2)

        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
        
        #Calculate population analysis and format all the data, then print to console
        bond_order_mulliken, charges_mulliken, total_charges_mulliken, bond_order_lowdin, charges_lowdin, total_charges_lowdin, bond_order_mayer, free_valences, total_valences = calculate_population_analysis(P, S, P_alpha - P_beta, ao_ranges, atoms, charges)
        space, space2, space3, charges_mulliken, charges_lowdin, free_valences, total_valences, atoms_formatted = format_population_analysis_output(charges_mulliken, charges_lowdin, total_charges_mulliken, bond_order_mulliken, bond_order_lowdin, free_valences, total_valences, atoms)


        log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
        log("      Mulliken Charges                Lowdin Charges              Mayer Free, Bonded, Total Valence", calculation, 2)
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
        log(f"  {atoms_formatted[0]}  {charges_mulliken[0]}                 {atoms_formatted[0]}  {charges_lowdin[0]}                 {atoms_formatted[0]}   {free_valences[0]},  {bond_order_mayer:.5f}, {total_valences[0]}", calculation, 2)
        log(f"  {atoms_formatted[1]}  {charges_mulliken[1]}                 {atoms_formatted[1]}  {charges_lowdin[1]}                 {atoms_formatted[1]}   {free_valences[1]},  {bond_order_mayer:.5f}, {total_valences[1]}", calculation, 2)
        log(f"\n  Sum of charges: {total_charges_mulliken:.5f}   {space}   Sum of charges: {total_charges_lowdin:.5f}", calculation, 2) 
        log(f"  Bond order: {bond_order_mulliken:.5f}      {space2}    Bond order: {bond_order_lowdin:.5f}      {space3}     Bond order: {bond_order_mayer:.5f}", calculation, 2) 
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)