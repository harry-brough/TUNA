import numpy as np
import tuna_scf as scf
import sys
import tuna_util as util
import tuna_dispersion as disp
import tuna_integral as integ
import tuna_postscf as postscf
import tuna_basis as basis_sets


def calculate_nuclear_repulsion(Z_list, coordinates): return np.prod(Z_list) / np.linalg.norm(coordinates[1] - coordinates[0])
    

def calculate_spin_contamination(P_alpha, P_beta, n_alpha, n_beta, S):

    s_squared_exact = ((n_alpha - n_beta) / 2) * ((n_alpha - n_beta) / 2 + 1)

    spin_contamination = n_beta - np.einsum("ii->", P_alpha.T @ S @ P_beta.T @ S, optimize=True)
    
    s_squared = s_squared_exact + spin_contamination

    return s_squared, s_squared_exact, spin_contamination




    
def rotate_molecular_orbitals(molecular_orbitals, n_occ, H_core, theta):

    homo_index = n_occ - 1
    lumo_index = n_occ

    rotation_matrix = np.eye(H_core.shape[0])

    rotation_matrix[homo_index:lumo_index + 1, homo_index:lumo_index + 1] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
    
    rotated_molecular_orbitals = molecular_orbitals @ rotation_matrix

    return rotated_molecular_orbitals



def setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_electrons_per_orbital, point_group, n_electrons, n_alpha, n_beta, rotate_guess_mos, norotate_guess_mos, silent=False):

    H_core = T + V_NE
    guess_epsilons = []; guess_mos = []
    E_guess = 0

    if reference == "RHF":

        if P_guess is not None and not silent: print("\n Using density matrix from previous step for guess. \n")

        else:
            
            if not silent: print(" Calculating one-electron density for guess...  ",end="")

            guess_epsilons, guess_mos = scf.diagonalise_Fock_matrix(H_core, X)
            P_guess = scf.construct_density_matrix(guess_mos, n_doubly_occ, n_electrons_per_orbital)

            E_guess = guess_epsilons[0]       

            if not silent: print("[Done]\n")


    elif reference == "UHF":    

        if P_guess_alpha is not None and P_guess_beta is not None and not silent: print("\n Using density matrices from previous step for guess. \n")

        else:
            
            if not silent: print(" Calculating one-electron density for guess...  ",end="")

            if point_group == "Dinfh" and n_electrons % 2 == 0 and not(norotate_guess_mos): rotate_guess_mos = True

            guess_epsilons, guess_mos = scf.diagonalise_Fock_matrix(H_core, X)

            guess_mos_alpha = rotate_molecular_orbitals(guess_mos, n_alpha, H_core, np.pi / 4) if rotate_guess_mos else guess_mos
                
            P_guess_alpha = scf.construct_density_matrix(guess_mos_alpha, n_alpha, n_electrons_per_orbital)
            P_guess_beta = scf.construct_density_matrix(guess_mos, n_beta, n_electrons_per_orbital)

            E_guess = guess_epsilons[0]
            P_guess = P_guess_alpha + P_guess_beta

            if not silent: print("[Done]\n")

        if rotate_guess_mos and not silent: print(" Initial guess density uses rotated molecular orbitals.\n")


    return E_guess, P_guess, P_guess_alpha, P_guess_beta, guess_epsilons, guess_mos




def calculate_fock_transformation_matrix(S):
        
    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    
    X = np.linalg.inv(S_sqrt)

    return X



def calculate_energy(calculation, atoms, coordinates, P_guess=None, P_guess_alpha=None, P_guess_beta=None, E_guess=None, terse=False, silent=False):

    if not silent: print("\n Setting up molecule...  ",end=""); sys.stdout.flush()

    molecule = util.Molecule(atoms, coordinates, calculation)
    
    atoms = molecule.atoms
    n_electrons = molecule.n_electrons
    coordinates = molecule.coordinates
    multiplicity = molecule.multiplicity
    reference = calculation.reference
    n_alpha = molecule.n_alpha
    n_beta = molecule.n_beta
    Z_list = molecule.Z_list
    point_group = molecule.point_group
    n_doubly_occ = molecule.n_doubly_occ
    method = calculation.method
    masses = molecule.masses
    if len(atoms) == 2: bond_length = molecule.bond_length
    n_electrons_per_orbital = calculation.n_electrons_per_orbital

    centre_of_mass = postscf.calculate_centre_of_mass(masses, coordinates)

    if calculation.decontract: atomic_orbitals = [[pg] for pg in molecule.pgs]
    else: atomic_orbitals = molecule.atomic_orbitals




    if not silent: 

        print("[Done]\n")

        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("   Molecule and Basis Information")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("  Molecular structure: " + molecule.molecular_structure)
        print("  Number of atoms: " + str(len(atoms)))
        print("  Number of basis functions: " + str(len(atomic_orbitals)))
        print("  Number of primitive Gaussians: " + str(len(molecule.pgs)))
        print("  Charge: " + str(molecule.charge))
        print("  Multiplicity: " + str(molecule.multiplicity))
        print("  Number of electrons: " + str(n_electrons))
        print(f"  Point group: {molecule.point_group}")
        if len(atoms) == 2: print(f"  Bond length: {util.bohr_to_angstrom(bond_length):.4f} ")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")



    if len(Z_list) == 2 and "X" not in atoms:

        if not silent: print(" Calculating nuclear repulsion energy...  ",end="")
        V_NN = calculate_nuclear_repulsion(Z_list, coordinates)

        if not silent: 
            
            print("[Done]\n"); 
            print(f" Nuclear repulsion energy: {V_NN:.10f}\n")
        
        if calculation.d2:     
            if not silent: print(" Calculating semi-empirical dispersion energy...  ",end="")
            E_D2 = disp.calculate_d2_energy(atoms, bond_length)

            if not silent: 
                
                print("[Done]"); 
                print(f" Dispersion energy (D2): {E_D2:.10f}\n")
            
        else: E_D2 = 0
        
    else: V_NN = 0; E_D2 = 0
        

    if n_electrons % 2 != 0 and calculation.reference == "RHF": util.error("Restricted Hartree-Fock is not compatible with an odd number of electrons!")
    if multiplicity != 1 and calculation.reference == "RHF": util.error("Restricted Hartree-Fock is not compatible non-singlet states!")

    if n_electrons == 0: 
    
        if not silent: util.warning("Calculation specified with zero electrons!"); print(f"Final energy: {V_NN:.10f}")
        
        util.finish_calculation(calculation)
        
    elif n_electrons < 0: util.error("Negative number of electrons specified!")
 

    if n_electrons > 1:
    
        if not silent: print(" Calculating one and two-electron integrals...  ",end=""); sys.stdout.flush()
        S, T, V_NE, D, V_EE = integ.evaluate_integrals(atomic_orbitals, np.array(Z_list, dtype=np.float64), coordinates, centre_of_mass)
        if not silent: print("[Done]")

        if not silent: print(" Constructing Fock transformation matrix...     ",end="")
        X = calculate_fock_transformation_matrix(S)
        if not silent: print("[Done]")


        E_guess, P_guess, P_guess_alpha, P_guess_beta, guess_epsilons, guess_mos = setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_electrons_per_orbital, point_group, n_electrons, n_alpha, n_beta, calculation.rotate_guess, calculation.norotate_guess, silent=silent)


        if not silent: 
            
            print(" Beginning self-consistent field cycle...\n")

            print(f" Using \"{calculation.scf_conv.get("word")}\" convergence criteria.")

            if calculation.diis and not calculation.damping: print(" Using DIIS for convergence acceleration.")
            elif calculation.diis and calculation.damping: print(" Using initial dynamic damping and DIIS for convergence acceleration.")
            elif calculation.damping and not calculation.slowconv and not calculation.veryslowconv: print(" Using permanent dynamic damping for convergence acceleration.")  
            if calculation.slowconv: print(" Using strong static damping for convergence acceleration.")  
            elif calculation.veryslowconv: print(" Using very strong static damping for convergence acceleration.")  
            if calculation.level_shift: print(" Using level shift for convergence acceleration.")
            if not calculation.diis and not calculation.damping and not calculation.level_shift: print(" No convergence acceleration used.")

            print("")

        final_energy, molecular_orbitals, epsilons, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy, P, P_alpha, P_beta = scf.SCF(molecule, calculation, T, V_NE, V_EE, V_NN, S, X, E_guess, P=P_guess, P_alpha=P_guess_alpha, P_beta=P_guess_beta, silent=silent)


        if not silent: postscf.print_energy_components(nuclear_electron_energy, kinetic_energy, exchange_energy, coulomb_energy, V_NN)


        if method == "MP2" or method == "SCS-MP2":

            if not silent:

                print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("         MP2 Energy and Density Calculation ")
                print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            import tuna_mp2 as mp2
            
            P_HF_mo_basis = molecular_orbitals.T @ S @ P @ S @ molecular_orbitals

            occupied_mos = molecular_orbitals[:, :n_doubly_occ]
            virtual_mos = molecular_orbitals[:, n_doubly_occ:]
            
            occupied_epsilons = epsilons[:n_doubly_occ]
            virtual_epsilons = epsilons[n_doubly_occ:]

            V_EE_mo_basis = mp2.transform_ao_two_electron_integrals(V_EE, occupied_mos, virtual_mos,silent=silent)
   

            if method == "MP2": E_MP2, P_MP2_mo_basis = mp2.calculate_mp2_energy_and_density(occupied_epsilons, virtual_epsilons, V_EE_mo_basis, P_HF_mo_basis,silent=silent, terse=terse)
            elif method == "SCS-MP2": E_MP2, P_MP2_mo_basis = mp2.calculate_scs_mp2_energy_and_density(occupied_epsilons, virtual_epsilons, V_EE_mo_basis, P_HF_mo_basis,silent=silent, terse=terse)
        
            P_MP2 = molecular_orbitals @ P_MP2_mo_basis @ molecular_orbitals.T

            natural_orbital_occupancies = np.sort(np.linalg.eigh(P_MP2_mo_basis)[0])[::-1]
            sum_of_occupancies = np.sum(natural_orbital_occupancies)
            
            if not silent and not terse: 
                
                print("\n  Natural orbital occupancies: \n")

                for i in range(len(natural_orbital_occupancies)): print(f"    {i + 1}.   {natural_orbital_occupancies[i]:.10f}")
    
                print(f"\n  Sum of natural orbital occupancies: {sum_of_occupancies:.6f}")
                print(f"  Trace of density matrix:  {np.trace(P_MP2_mo_basis):.6f}")
        
                print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            elif terse: 
                print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            P = P_MP2

        else: E_MP2 = 0
        
        
    else: 

        if not silent: print(" Calculating one-electron integrals...    ",end=""); sys.stdout.flush()
        S, T, V_NE, D, V_EE = integ.evaluate_integrals(atomic_orbitals, np.array(Z_list, dtype=np.float64), coordinates, centre_of_mass, two_electron_ints=False)
        if not silent: print("[Done]")     

        if not silent: print(" Constructing Fock transformation matrix...  ",end="")
        X = calculate_fock_transformation_matrix(S)
        if not silent: print("[Done]")


        E_guess, P_guess, P_guess_alpha, P_guess_beta, epsilons, molecular_orbitals = setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_electrons_per_orbital, point_group, n_electrons, n_alpha, n_beta, calculation.rotate_guess, calculation.norotate_guess, silent=silent)

        final_energy = E_guess
        P = P_guess
        P_alpha = P_guess / 2
        P_beta = P_guess / 2
        E_MP2 = 0

        if method not in ["HF", "RHF", "UHF"]: util.warning("A correlated calculation has been requested on a one-electron system! Energy will be Hartree-Fock only.")

    ao_ranges = [len(basis_sets.generate_atomic_orbitals(atom, molecule.basis, coord)) for atom, coord in zip(atoms, coordinates)]
    

    if reference == "UHF": 
        
        if not silent and n_electrons > 1:

            s_squared, s_squared_exact, spin_contamination = calculate_spin_contamination(P_alpha, P_beta, n_alpha, n_beta, S)

            print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("             Spin Contamination       ")
            print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            print(f" Exact S^2 expectation value:      {s_squared_exact:.6f}")
            print(f" UHF S^2 expectation value:        {s_squared:.6f}")
            print(f"\n Spin contamination:               {spin_contamination:.6f}")

            print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if reference == "RHF": P_alpha = P /2; P_beta = P /2
    if not terse and not silent: postscf.post_scf_output(molecule, calculation, epsilons, molecular_orbitals, P, S, ao_ranges, D, P_alpha, P_beta)
    
    if not silent: 
        
        if calculation.reference == "RHF": print("\n Final restricted Hartree-Fock energy: " + f"{final_energy:.10f}")
        if calculation.reference == "UHF": print("\n Final unrestricted Hartree-Fock energy: " + f"{final_energy:.10f}")

    if calculation.d2:
    
        final_energy += E_D2

        if not silent: print(" Dispersion-corrected final energy: " + f"{final_energy:.10f}")
    

    if method == "MP2" or method == "SCS-MP2": 
    
        final_energy += E_MP2
        
        if not silent: print(f" Correlation energy from {method}: " + f"{E_MP2:.10f}\n")
        if not silent: print(" Final single point energy: " + f"{final_energy:.10f}")
    

    if calculation.densplot and not silent: postscf.construct_electron_density(P, 0.07, molecule)


    scf_output = util.Output(final_energy, P, S, ao_ranges, epsilons, molecular_orbitals, D, P_alpha, P_beta)

    return scf_output, molecule


    

def scan_coordinate(calculation, atoms, starting_coordinates):

    coordinates = util.bohr_to_angstrom(starting_coordinates)
    number_of_steps = calculation.scannumber
    step_size = calculation.scanstep

    print(f"Initialising a {number_of_steps} step coordinate scan in {step_size:.4f} Angstrom increments.") 
    print(f"Starting at a bond length of {np.linalg.norm(coordinates[1] - coordinates[0]):.4f} Angstroms.\n")
    
    bond_lengths = [] ;energies = []   
    P_guess = None; E_guess = None; P_guess_alpha = None; P_guess_beta = None


    for step in range(1, number_of_steps + 1):
        
        bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Starting scan step {step} of {number_of_steps} with bond length of {bond_length:.4f} Angstroms...")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        
        scf_output, molecule = calculate_energy(calculation, atoms, util.angstrom_to_bohr(coordinates), P_guess, P_guess_alpha, P_guess_beta, E_guess, terse=True)

        energy = scf_output.energy

        if calculation.moread: P_guess = scf_output.P; E_guess = energy; P_guess_alpha = scf_output.P_alpha; P_guess_beta = scf_output.P_beta
        else: P_guess = None; E_guess = None


        energies.append(energy)
        bond_lengths.append(bond_length)

        coordinates = np.array([coordinates[0], [0,0,coordinates[1][2] + step_size]])
        
    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    
    print("\nCoordinate scan calculation finished, printing energy values...\n")
    
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("   R (Angstroms)    Energy (Hartree)")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    for energy, bond_length in zip(energies,bond_lengths):

        if energy > 0: energy_f = " " + f"{energy:.10f}"
        else: energy_f = f"{energy:.10f}"

        print(f"      {bond_length:.4f}          {energy_f}")

    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    

    if calculation.scanplot:
        
        print("Plotting energy profile diagram...   ",end=""); sys.stdout.flush()
        
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(10,5))    
        plt.plot(bond_lengths, energies, color=(0.75,0,0),linewidth=1.75)
        plt.xlabel("Bond Length (Angstrom)", fontweight="bold", labelpad=10, fontfamily="Arial",fontsize=12)
        plt.ylabel("Energy (hartree)",labelpad=10, fontweight="bold", fontfamily="Arial",fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.25, length=6, direction='out')
        ax.tick_params(axis='both', which='minor', labelsize=11, width=1.25, length=3, direction='out')
        
        for spine in ax.spines.values(): spine.set_linewidth(1.25)
        
        plt.minorticks_on()
        plt.tight_layout() 
        print("[Done]")
        
        
        plt.show()
