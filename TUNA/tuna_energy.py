import numpy as np
import tuna_scf as scf
import sys
from tuna_util import *
import tuna_dispersion as disp
import tuna_integral as integ
import tuna_postscf as postscf
import tuna_mpn as mpn
import tuna_plot as plot


def calculate_nuclear_repulsion(charges, coordinates):
    
    """
    
    Requires list of nuclear charges (list) and coordinates (array).

    Returns nuclear repulsion energy in hartree.

    """
    
    return np.prod(charges) / np.linalg.norm(coordinates[1] - coordinates[0])
    


def calculate_spin_contamination(P_alpha, P_beta, n_alpha, n_beta, S):

    """
    
    Requires alpha and beta density matrices (arrays), number of alpha and beta electrons (int) and overlap matrix (array).

    Calculates exact S^2 and spin contamination.

    Returns S^2, theoretically exact S^2 and spin contamination.

    """

    #Theoretically exact S^2 from number of alpha and beta electrons
    s_squared_exact = ((n_alpha - n_beta) / 2) * ((n_alpha - n_beta) / 2 + 1)

    #Tensor contraction to calculate spin contamination
    spin_contamination = n_beta - np.einsum("ii->", P_alpha.T @ S @ P_beta.T @ S, optimize=True)
    
    s_squared = s_squared_exact + spin_contamination

    return s_squared, s_squared_exact, spin_contamination




def rotate_molecular_orbitals(molecular_orbitals, n_occ, H_core, theta):

    """
    
    Requires molecular orbitals (array), number of occupied orbitals (int), core Hamiltonian (array) and rotation angle theta (float).

    Creates a rotation matrix to rotate the HOMO and LUMO with each other by a requested angle theta, to break the symmetry in UHF. This
    can be used for either molecular orbitals of alpha and beta spin orbitals.

    Returns the rotated molecular orbitals (array).

    """
    
    homo_index = n_occ - 1
    lumo_index = n_occ

    dimension = H_core.shape[0]
    rotation_matrix = np.eye(dimension)

    #Makes sure there is a HOMO and a LUMO to rotate, builds rotation matrix using sine and cosine of the requested angle, at the HOMO and LUMO indices
    if dimension > 1:

        rotation_matrix[homo_index:lumo_index + 1, homo_index:lumo_index + 1] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])

    #Rotates molecularn orbitals with this matrix
    rotated_molecular_orbitals = molecular_orbitals @ rotation_matrix

    return rotated_molecular_orbitals




def setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_alpha, n_beta, rotate_guess_mos, norotate_guess_mos, calculation, silent=False):

    """
    
    Requires guess density matrix (array), alpha and beta guess densities (arrays), energy guess 

    Builds the core Hamiltonian from kinetic matrix and nuclear-electron attraction matrix. If the reference is RHF, checks if there's a density matrix supplied and if 
    not then calculates the one-electron density to make the guess, from the core Hamiltonian. For a UHF reference, if the alpha and beta guess density matrices are not
    supplied, the one-electron density is calculated similarly, and then the alpha density may be rotated to break the symmetry, if there are an even number of electrons.

    Returns energy guess (float), density guess (array), alpha and beta density guesses (arrays), guess epsilons (array) and guess MOs (array).
    
    """

    H_core = T + V_NE

    guess_epsilons = []; 
    guess_mos = []


    if reference == "RHF":
        
        #If there's a guess density, just use that
        if P_guess is not None and not silent: log("\n Using density matrix from previous step for guess. \n", calculation, 1)

        else:
            
            if not silent: log(" Calculating one-electron density for guess...  ", calculation, end="")

            #Diagonalise core Hamiltonian for one-electron guess, then build density matrix (2 electrons per orbital) from these guess molecular orbitals
            guess_epsilons, guess_mos = scf.diagonalise_Fock_matrix(H_core, X)
            P_guess = scf.construct_density_matrix(guess_mos, n_doubly_occ, 2)

            #Take lowest energy guess epsilon for guess energy
            E_guess = guess_epsilons[0]       

            if not silent: log("[Done]\n", calculation)


    elif reference == "UHF":    

        #If there's a guess density, just use that
        if P_guess_alpha is not None and P_guess_beta is not None and not silent: log("\n Using density matrices from previous step for guess. \n", calculation)

        else:
            
            if not silent: log(" Calculating one-electron density for guess...  ", calculation, end="")

            #Only rotate guess MOs if there's an even number of electrons, and it hasn't been overridden by NOROTATE
            if (n_alpha + n_beta) % 2 == 0 and not norotate_guess_mos: rotate_guess_mos = True

            #Diagonalise core Hamiltonian for one-electron guess, then build density matrix from these guess molecular orbitals
            guess_epsilons, guess_mos = scf.diagonalise_Fock_matrix(H_core, X)

            #Rotate the alpha MOs if this is requested, otherwise take the alpha guess to equal the beta guess
            theta = calculation.theta
            guess_mos_alpha = rotate_molecular_orbitals(guess_mos, n_alpha, H_core, theta) if rotate_guess_mos else guess_mos

            #Construct density matrices (1 electron per orbital) for the alpha and beta guesses
            P_guess_alpha = scf.construct_density_matrix(guess_mos_alpha, n_alpha, 1)
            P_guess_beta = scf.construct_density_matrix(guess_mos, n_beta, 1)

            #Take lowest energy guess epsilon for guess energy
            E_guess = guess_epsilons[0]

            #Add together alpha and beta densities for total density
            P_guess = P_guess_alpha + P_guess_beta

            if not silent: log("[Done]\n", calculation)

        if rotate_guess_mos and not silent: log(" Initial guess density uses rotated molecular orbitals.\n", calculation)


    return E_guess, P_guess, P_guess_alpha, P_guess_beta, guess_epsilons, guess_mos




def calculate_fock_transformation_matrix(S):

    """
    
    Requires overlap matrix (array).

    Diagonalises the overlap matrix for its eigenvectors and eigenvalues, the finds the square root of the overlap matrices with these
    before finding the inverse of this for the Fock transformation matrix.

    Returns the Fock transformation matrix, X (array).

    """
        
    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    
    X = np.linalg.inv(S_sqrt)

    return X




def calculate_energy(calculation, atoms, coordinates, P_guess=None, P_guess_alpha=None, P_guess_beta=None, E_guess=None, terse=False, silent=False):

    """
    
    Requires calculation (Calculation), atoms (list), coordinates (array) and optional arguments, including guess density matrices (arrays) and
    energy (float), whether postscf information should be printed (terse, bool) and whether anything should be printed (silent, bool) as in
    gradient calculations.
    
    Creates a Molecule object, extracts useful quantities then calculates the energy, defined by the user's method. Prints various molecular
    information. Depending on the number of electrons, calculates different quantities. For multi-electron calculations, an initial guess
    is made, the SCF cycle begins, a correlated calculation is called if requested and various information is printed to the console.

    Returns SCF output (Output), molecule (Molecule), final energy (float), and final density matrix (array).

    """

    if not silent: log("\n Setting up molecule...  ", calculation, 1, end=""); sys.stdout.flush()

    #Builds molecule object using calculation and atomic parameters
    molecule = Molecule(atoms, coordinates, calculation)
    
    #Unpacking of various useful calculation quantities
    method = calculation.method
    reference = calculation.reference

    #Unpacking of various useful molecular properties
    atoms = molecule.atoms
    charges = molecule.charges
    coordinates = molecule.coordinates
    bond_length = molecule.bond_length
    centre_of_mass = molecule.centre_of_mass
    atomic_orbitals = molecule.atomic_orbitals
    n_doubly_occ = molecule.n_doubly_occ
    n_electrons = molecule.n_electrons
    n_alpha = molecule.n_alpha
    n_beta = molecule.n_beta


    #The silent boolean is activated eg. during a numerical gradient calculation
    if not silent: 

        log("[Done]\n", calculation, 1)

        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        log("   Molecule and Basis Information", calculation, 1)
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        log("  Molecular structure: " + molecule.molecular_structure, calculation, 1)
        log("  Number of atoms: " + str(len(atoms)), calculation, 1)
        log("  Number of basis functions: " + str(len(atomic_orbitals)), calculation, 1)
        log("  Number of primitive Gaussians: " + str(len(molecule.pgs)), calculation, 1)
        log("  Charge: " + str(molecule.charge), calculation, 1)
        log("  Multiplicity: " + str(molecule.multiplicity), calculation, 1)
        log("  Number of electrons: " + str(n_electrons), calculation, 1)
        log(f"  Point group: {molecule.point_group}", calculation, 1)
        if len(atoms) == 2: log(f"  Bond length: {bohr_to_angstrom(bond_length):.4f} ", calculation, 1)
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, 1)


    #Nuclear repulsion and dispersion energy are only calculated if there are two, non ghost, atoms present
    if len(charges) == 2 and not any("X" in atom for atom in atoms):

        #Calculates nuclear repulsion energy
        if not silent: log(" Calculating nuclear repulsion energy...  ", calculation, 1, end="")
        V_NN = calculate_nuclear_repulsion(charges, coordinates)
        if not silent: log(f"[Done]\n\n Nuclear repulsion energy: {V_NN:.10f}\n", calculation, 1)
        
        #Calculates D2 dispersion energy if requested
        if calculation.d2:  

            if not silent: log(" Calculating semi-empirical dispersion energy...  ", calculation, 1, end="")
            E_D2 = disp.calculate_d2_energy(atoms, bond_length)
            if not silent: log(f"[Done]\n\n Dispersion energy (D2): {E_D2:.10f}\n", calculation, 1)
            
        else: E_D2 = 0
        
    else: V_NN = 0; E_D2 = 0
        

    if n_electrons < 0: error("Negative number of electrons specified!")

    elif n_electrons == 0: 

        #If zero electrons are specified, the only energy is due to nuclear repulsion, which is printed and the calculation ends
        warning("Calculation specified with zero electrons!\n")
        log(f"Final energy: {V_NN:.10f}", calculation, 1)
        
        finish_calculation(calculation)
        

    elif n_electrons == 1: 

        if method not in ["HF", "RHF", "UHF"]: error("A correlated calculation has been requested on a one-electron system!")

        #Calculates one-electron integrals
        if not silent: log(" Calculating one-electron integrals...    ", calculation, 1, end=""); sys.stdout.flush()
        S, T, V_NE, D, V_EE = integ.evaluate_integrals(atomic_orbitals, np.array(charges, dtype=np.float64), coordinates, centre_of_mass, two_electron_ints=False)
        if not silent: log("[Done]", calculation, 1)     

        #Calculates Fock transformation matrix from overlap matrix
        if not silent: log(" Constructing Fock transformation matrix...  ", calculation, 1, end="")
        X = calculate_fock_transformation_matrix(S)
        if not silent: log("[Done]", calculation, 1)
        
        #No SCF, so no reading in orbitals

        P_guess = None
        P_guess_alpha = None
        P_guess_beta = None
        E_guess = None

        #Builds initial guess, which is the final answer for the one-electron case
        E_guess, P_guess, P_guess_alpha, P_guess_beta, epsilons, molecular_orbitals = setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_alpha, n_beta, calculation.rotate_guess, calculation.norotate_guess, calculation, silent=silent)

        final_energy = E_guess + V_NN
        P = P_guess
        P_alpha = P_guess / 2
        P_beta = P_guess / 2
        E_MP2 = 0
        E_MP3 = 0

        #Builds energy output object with all the calculated quantities from the one-electron guess
        scf_output = Output(final_energy, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals, None, epsilons, epsilons, None)

        #No beta electrons for the one-electron case exists
        epsilons_alpha = epsilons
        epsilons_beta = None
        molecular_orbitals_alpha = molecular_orbitals
        molecular_orbitals_beta = None


    else:

        #Calculates one- and two-electron integrals
        if not silent: log(" Calculating one- and two-electron integrals...  ", calculation, 1, end=""); sys.stdout.flush()
        S, T, V_NE, D, V_EE = integ.evaluate_integrals(atomic_orbitals, np.array(charges, dtype=np.float64), coordinates, centre_of_mass)
        if not silent: log("[Done]", calculation, 1)

        #Calculates Fock transformation matrix from overlap matrix
        if not silent: log(" Constructing Fock transformation matrix...     ", calculation, 1, end="")
        X = calculate_fock_transformation_matrix(S)
        if not silent: log("[Done]", calculation, 1)

        #Calculates one-electron density for initial guess
        E_guess, P_guess, P_guess_alpha, P_guess_beta, _, _ = setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_alpha, n_beta, calculation.rotate_guess, calculation.norotate_guess, calculation, silent=silent)

        if not silent: 
            
            log(" Beginning self-consistent field cycle...\n", calculation, 1)

            #Prints convergence criteria specified
            log(f" Using \"{calculation.scf_conv.get("word")}\" convergence criteria.", calculation, 1)

            #Prints the chosen SCF convergence acceleration options
            if calculation.diis and not calculation.damping: log(" Using DIIS for convergence acceleration.", calculation, 1)
            elif calculation.diis and calculation.damping: log(" Using initial dynamic damping and DIIS for convergence acceleration.", calculation, 1)
            elif calculation.damping and not calculation.slowconv and not calculation.veryslowconv: log(" Using permanent dynamic damping for convergence acceleration.", calculation, 1)  
            if calculation.slowconv: log(" Using strong static damping for convergence acceleration.", calculation, 1)  
            elif calculation.veryslowconv: log(" Using very strong static damping for convergence acceleration.", calculation, 1)  
            if calculation.level_shift: log(f" Using level shift for convergence acceleration with parameter {calculation.level_shift_parameter:.2f}.", calculation, 1)
            if not calculation.diis and not calculation.damping and not calculation.level_shift: log(" No convergence acceleration used.", calculation, 1)

            log("", calculation, 1)

        #Starts SCF cycle for two-electron energy
        scf_output, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy = scf.SCF(molecule, calculation, T, V_NE, V_EE, V_NN, S, X, E_guess, P=P_guess, P_alpha=P_guess_alpha, P_beta=P_guess_beta, silent=silent)

        #Extracts useful quantities from SCF output object
        molecular_orbitals = scf_output.molecular_orbitals
        molecular_orbitals_alpha = scf_output.molecular_orbitals_alpha  
        molecular_orbitals_beta = scf_output.molecular_orbitals_beta   
        epsilons = scf_output.epsilons
        epsilons_alpha = scf_output.epsilons_alpha
        epsilons_beta = scf_output.epsilons_beta
        P = scf_output.P
        P_alpha = scf_output.P_alpha
        P_beta = scf_output.P_beta
        final_energy = scf_output.energy

        #Packs dipole integrals into SCF output object
        scf_output.D = D

        if reference == "UHF" and not silent: 
            
            #Calculates UHF spin contamination and prints to the console
            s_squared, s_squared_exact, spin_contamination = calculate_spin_contamination(P_alpha, P_beta, n_alpha, n_beta, S)

            log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
            log("             UHF Spin Contamination       ", calculation, 2)
            log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)

            log(f"  Exact S^2 expectation value:     {s_squared_exact:.6f}", calculation, 2)
            log(f"  UHF S^2 expectation value:       {s_squared:.6f}", calculation, 2)
            log(f"\n  Spin contamination:              {spin_contamination:.6f}", calculation, 2)

            log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)


        #Prints the individual components of the total SCF energy
        if not silent: postscf.print_energy_components(nuclear_electron_energy, kinetic_energy, exchange_energy, coulomb_energy, V_NN, calculation)

        #If a correlated calculation is requested, calculates the energy and density matrix
        if method in ["MP2", "UMP2", "SCS-MP2", "MP3", "UMP3", "SCS-MP3"]: E_MP2, E_MP3, P = mpn.calculate_moller_plesset(calculation.mp2_basis, method, molecule, scf_output, V_EE, calculation, silent=silent, terse=terse)

    #Prints post SCF information, as long as its not an optimisation that hasn't finished yet
    if not terse and not silent: postscf.post_scf_output(molecule, calculation, epsilons, molecular_orbitals, P, S, molecule.ao_ranges, D, P_alpha, P_beta, epsilons_alpha, epsilons_beta, molecular_orbitals_alpha, molecular_orbitals_beta)
    
    if not silent: 
        
        if reference == "RHF": log("\n Final restricted Hartree-Fock energy: " + f"{final_energy:.10f}", calculation, 1)
        else: log("\n Final unrestricted Hartree-Fock energy: " + f"{final_energy:.10f}", calculation, 1)


    #Adds up and prints MP2 energies
    if method in ["MP2", "SCS-MP2", "UMP2"]: 
    
        final_energy += E_MP2
        
        if not silent: 

            log(f" Correlation energy from {method}: " + f"{E_MP2:.10f}\n", calculation, 1)
            log(" Final single point energy: " + f"{final_energy:.10f}", calculation, 1)
    

    #Adds up and prints MP3 energies
    elif method in ["MP3", "UMP3", "SCS-MP3"]:
        
        final_energy += E_MP2 + E_MP3

        if not silent:

            if method == "SCS-MP3":

                log(f" Correlation energy from SCS-MP2: " + f"{E_MP2:.10f}", calculation, 1)
                log(f" Correlation energy from SCS-MP3: " + f"{E_MP3:.10f}\n", calculation, 1)

            else:

                log(f" Correlation energy from MP2: " + f"{E_MP2:.10f}", calculation, 1)
                log(f" Correlation energy from MP3: " + f"{E_MP3:.10f}\n", calculation, 1)

            log(" Final single point energy: " + f"{final_energy:.10f}", calculation, 1)


    #Adds on D2 energy, and prints this as dispersion-corrected final energy
    if calculation.d2:
    
        final_energy += E_D2

        if not silent: log("\n Semi-empirical dispersion energy: " + f"{E_D2:.10f}", calculation, 1)
        if not silent: log(" Dispersion-corrected final energy: " + f"{final_energy:.10f}", calculation, 1)
    
    #Calculates and plots electron density if this is requested with DENSPLOT keyword
    if calculation.densplot and not silent: plot.construct_electron_density(P, 0.07, molecule, calculation)

    return scf_output, molecule, final_energy, P


    

def scan_coordinate(calculation, atoms, starting_coordinates):

    """
    
    Requires calculation (Calculation), atoms (list), starting coordinates for scan (array).

    Loops through a number of scan steps (determined by SCANNUMBER), with a certain incrementing bond length (SCANSTEP) between
    each step. Calculates the energy at each point, prints a table of bond length against energy, and a matplotlib graph
    if this is requested with the SCANPLOT keyword.
    
    """

    #Converts coordinates into angstrom, for printing
    coordinates = bohr_to_angstrom(starting_coordinates)

    #Unpacks useful quantities
    number_of_steps = calculation.scannumber
    step_size = calculation.scanstep

    log(f"Initialising a {number_of_steps} step coordinate scan in {step_size:.4f} Angstrom increments.", calculation, 1) 
    log(f"Starting at a bond length of {np.linalg.norm(coordinates[1] - coordinates[0]):.4f} Angstroms.\n", calculation, 1)
    
    #Initialises lists and guess values
    bond_lengths = [] ; energies = []   
    P_guess = None; E_guess = None; P_guess_alpha = None; P_guess_beta = None


    for step in range(1, number_of_steps + 1):
        
        #Calculates bond length before construction of Molecule to print this out
        bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

        log("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        log(f"Starting scan step {step} of {number_of_steps} with bond length of {bond_length:.4f} angstroms...", calculation, 1)
        log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, 1)
        
        #Calculates the energy at the coordinates (in bohr) specified
        scf_output, _, energy, _ = calculate_energy(calculation, atoms, angstrom_to_bohr(coordinates), P_guess, P_guess_alpha, P_guess_beta, E_guess, terse=True)

        #If MOREAD keyword is used, then the energy and densities are used for the next calculation
        if calculation.moread: P_guess = scf_output.P; E_guess = energy; P_guess_alpha = scf_output.P_alpha; P_guess_beta = scf_output.P_beta
        else: P_guess = None; E_guess = None

        #Appends energies and bond lengths to lists
        energies.append(energy)
        bond_lengths.append(bond_length)

        #Builds new coordinates by adding step size on
        coordinates = np.array([coordinates[0], [0,0,coordinates[1][2] + step_size]])
        
    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)    
    
    log("\nCoordinate scan calculation finished, logging energy values...\n", calculation, 1)
    
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log("   R (angstroms)    Energy (hartree)", calculation, 1)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)

    #Prints a table of bond lengths and corresponding energies
    for energy, bond_length in zip(energies, bond_lengths):
        
        #Formats energy to align the decimal points
        if energy > 0: energy_f = " " + f"{energy:.10f}"
        else: energy_f = f"{energy:.10f}"

        log(f"      {bond_length:.4f}          {energy_f}", calculation, 1)

    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, 1)
    
    #If SCANPLOT keyword is used, plots and shows a matplotlib graph of the table data
    if calculation.scanplot: plot.scan_plot(calculation, bond_lengths, energies)

