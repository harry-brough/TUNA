import numpy as np
import tuna_scf as scf
import sys
import tuna_util as util
import tuna_dispersion as disp
import tuna_integral as integ
import tuna_postscf as postscf
import tuna_mpn as mpn
    

def calculate_nuclear_repulsion(Z_list, coordinates): return np.prod(Z_list) / np.linalg.norm(coordinates[1] - coordinates[0])
    

def calculate_spin_contamination(P_alpha, P_beta, n_alpha, n_beta, S):

    s_squared_exact = ((n_alpha - n_beta) / 2) * ((n_alpha - n_beta) / 2 + 1)

    spin_contamination = n_beta - np.einsum("ii->", P_alpha.T @ S @ P_beta.T @ S, optimize=True)
    
    s_squared = s_squared_exact + spin_contamination

    return s_squared, s_squared_exact, spin_contamination




def rotate_molecular_orbitals(molecular_orbitals, n_occ, H_core, theta):

    homo_index = n_occ - 1
    lumo_index = n_occ

    dimension = H_core.shape[0]
    rotation_matrix = np.eye(dimension)

    if dimension > 1:

        rotation_matrix[homo_index:lumo_index + 1, homo_index:lumo_index + 1] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
        
    rotated_molecular_orbitals = molecular_orbitals @ rotation_matrix

    return rotated_molecular_orbitals



def setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_electrons_per_orbital, point_group, n_electrons, n_alpha, n_beta, rotate_guess_mos, norotate_guess_mos, calculation, silent=False):

    H_core = T + V_NE
    guess_epsilons = []; guess_mos = []
    E_guess = 0

    if reference == "RHF":

        if P_guess is not None and not silent: util.log("\n Using density matrix from previous step for guess. \n", calculation)

        else:
            
            if not silent: util.log(" Calculating one-electron density for guess...  ", calculation, end="")

            guess_epsilons, guess_mos = scf.diagonalise_Fock_matrix(H_core, X)
            P_guess = scf.construct_density_matrix(guess_mos, n_doubly_occ, n_electrons_per_orbital)

            E_guess = guess_epsilons[0]       

            if not silent: util.log("[Done]\n", calculation)


    elif reference == "UHF":    

        if P_guess_alpha is not None and P_guess_beta is not None and not silent: util.log("\n Using density matrices from previous step for guess. \n", calculation)

        else:
            
            if not silent: util.log(" Calculating one-electron density for guess...  ", calculation, end="")

            if n_electrons % 2 == 0 and not(norotate_guess_mos): rotate_guess_mos = True

            guess_epsilons, guess_mos = scf.diagonalise_Fock_matrix(H_core, X)

            theta = calculation.theta
            guess_mos_alpha = rotate_molecular_orbitals(guess_mos, n_alpha, H_core, theta) if rotate_guess_mos else guess_mos
                
            P_guess_alpha = scf.construct_density_matrix(guess_mos_alpha, n_alpha, n_electrons_per_orbital)
            P_guess_beta = scf.construct_density_matrix(guess_mos, n_beta, n_electrons_per_orbital)

            E_guess = guess_epsilons[0]
            P_guess = P_guess_alpha + P_guess_beta

            if not silent: util.log("[Done]\n", calculation)

        if rotate_guess_mos and not silent: util.log(" Initial guess density uses rotated molecular orbitals.\n", calculation)


    return E_guess, P_guess, P_guess_alpha, P_guess_beta, guess_epsilons, guess_mos




def calculate_fock_transformation_matrix(S):
        
    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    
    X = np.linalg.inv(S_sqrt)

    return X




def calculate_moller_plesset(mp2_type, method, molecule, scf_output, V_EE_ao_basis, silent=False, terse=False):

    S = scf_output.S
    P = scf_output.P

    E_MP2 = 0
    E_MP3 = 0


    if mp2_type == "MO":

        n_doubly_occ = molecule.n_doubly_occ
        epsilons = scf_output.epsilons
        molecular_orbitals = scf_output.molecular_orbitals
    
        if not silent:

            print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("         MP2 Energy and Density Calculation ")
            print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


        P_HF_mo_basis = molecular_orbitals.T @ S @ P @ S @ molecular_orbitals

        occupied_mos = molecular_orbitals[:, :n_doubly_occ]
        virtual_mos = molecular_orbitals[:, n_doubly_occ:]
        
        occupied_epsilons = epsilons[:n_doubly_occ]
        virtual_epsilons = epsilons[n_doubly_occ:]

        V_EE_mo_basis = mpn.transform_ao_two_electron_integrals(V_EE_ao_basis, occupied_mos, virtual_mos,silent=silent)


        if method == "MP2": E_MP2, P_MP2_mo_basis = mpn.calculate_mp2_energy_and_density(occupied_epsilons, virtual_epsilons, V_EE_mo_basis, P_HF_mo_basis,silent=silent, terse=terse)
        elif method in ["SCS-MP2", "SCS-MP3"]: 
            
            E_MP2, P_MP2_mo_basis = mpn.calculate_scs_mp2_energy_and_density(occupied_epsilons, virtual_epsilons, V_EE_mo_basis, P_HF_mo_basis,silent=silent, terse=terse)
        
            if not silent: util.warning("Density is not spin-component-scaled!", 2)

        P_MP2 = molecular_orbitals @ P_MP2_mo_basis @ molecular_orbitals.T

        natural_orbital_occupancies = np.sort(np.linalg.eigh(P_MP2_mo_basis)[0])[::-1]
        sum_of_occupancies = np.sum(natural_orbital_occupancies)
        
        if not silent: 
            
            print("\n  Natural orbital occupancies: \n")

            for i in range(len(natural_orbital_occupancies)): print(f"    {i + 1}.   {natural_orbital_occupancies[i]:.10f}")

            print(f"\n  Sum of natural orbital occupancies: {sum_of_occupancies:.6f}")
            print(f"  Trace of density matrix:  {np.trace(P_MP2_mo_basis):.6f}")

        if not silent: print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if method == "SCS-MP3": 
            
            mp2_type = "SO"
            scs_mp2_energy = E_MP2

        elif terse: print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        P = P_MP2


    if mp2_type == "SO":

        molecular_orbitals_alpha = scf_output.molecular_orbitals_alpha

        if molecule.n_beta == 0: 
            
            epsilons_combined = scf_output.epsilons_alpha
            molecular_orbitals_beta = molecular_orbitals_alpha

        else: 
            
            epsilons_combined = scf_output.epsilons_combined
            molecular_orbitals_beta = scf_output.molecular_orbitals_beta


        n_occ = molecule.n_occ


        if not silent:

            print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if method in ["MP2", "UMP2"]: print("                MP2 Energy Calculation ")
            elif method in ["MP3", "UMP3", "SCS-MP3"]: print("                MP3 Energy Calculation ")

            print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
 

        V_EE_spin_block, C_spin_block = mpn.spin_block_two_electron_integrals(V_EE_ao_basis, molecular_orbitals_alpha, molecular_orbitals_beta, silent=silent)

        C_spin_block = C_spin_block[:, epsilons_combined.argsort()] 

        epsilons_sorted = np.sort(epsilons_combined)

        so_two_electron_integrals = mpn.transform_spin_blocked_two_electron_integrals(V_EE_spin_block, C_spin_block, silent=silent)

        o = slice(None, n_occ)
        v = slice(n_occ, None)

        e_ijab = mpn.build_epsilons_tensor(epsilons_sorted, o, v)

        if method != "SCS-MP3": E_MP2 = mpn.calculate_spin_orbital_MP2_energy(e_ijab, so_two_electron_integrals, o, v, silent=silent)

        if method in ["MP3", "UMP3", "SCS-MP3"]: E_MP3 = mpn.calculate_spin_orbital_MP3_energy(e_ijab, so_two_electron_integrals, o, v, silent=silent)

        if method == "SCS-MP3":

            mp3_scaling = 1 / 4
            E_MP3 *= mp3_scaling
            
            if not silent: 
                
                print(f"  Scaling for MP3: {mp3_scaling}\n")
                print(f"  Scaled MP3 correlation energy: {E_MP3:.10f}")
                print(f"  SCS-MP3 correlation energy: {(E_MP3 + scs_mp2_energy):.10f}")
        

        if not silent: print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



    return E_MP2, E_MP3, P





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

    if len(atoms) == 2 and "XH" not in atoms and "XHE" not in atoms: centre_of_mass = postscf.calculate_centre_of_mass(masses, coordinates)
    else: centre_of_mass = 0

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

        if not silent: util.log(" Calculating nuclear repulsion energy...  ", calculation, end="")
        V_NN = calculate_nuclear_repulsion(Z_list, coordinates)

        if not silent: 
            
            util.log("[Done]\n", calculation)
            util.log(f" Nuclear repulsion energy: {V_NN:.10f}\n", calculation)
        
        if calculation.d2:     
            if not silent: print(" Calculating semi-empirical dispersion energy...  ",end="")
            E_D2 = disp.calculate_d2_energy(atoms, bond_length)

            if not silent: 
                
                print("[Done]"); 
                print(f" Dispersion energy (D2): {E_D2:.10f}\n")
            
        else: E_D2 = 0
        
    else: V_NN = 0; E_D2 = 0
        

    if n_electrons % 2 != 0 and calculation.reference == "RHF" and n_electrons != 1: util.error("Restricted Hartree-Fock is not compatible with an odd number of electrons!")
    if multiplicity != 1 and calculation.reference == "RHF" and n_electrons != 1: util.error("Restricted Hartree-Fock is not compatible non-singlet states!")

    if n_electrons < 0: util.error("Negative number of electrons specified!")

    elif n_electrons == 0: 
    
        if not silent: util.warning("Calculation specified with zero electrons!\n"); util.log(f"Final energy: {V_NN:.10f}", calculation)
        
        util.finish_calculation(calculation)
        

    elif n_electrons == 1: 

        if not silent: util.log(" Calculating one-electron integrals...    ", calculation, end=""); sys.stdout.flush()
        S, T, V_NE, D, V_EE = integ.evaluate_integrals(atomic_orbitals, np.array(Z_list, dtype=np.float64), coordinates, centre_of_mass, two_electron_ints=False)
        if not silent: util.log("[Done]", calculation)     

        if not silent: util.log(" Constructing Fock transformation matrix...  ", calculation, end="")
        X = calculate_fock_transformation_matrix(S)
        if not silent: util.log("[Done]", calculation)


        E_guess, P_guess, P_guess_alpha, P_guess_beta, epsilons, molecular_orbitals = setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_electrons_per_orbital, point_group, n_electrons, n_alpha, n_beta, calculation.rotate_guess, calculation.norotate_guess, calculation, silent=silent)

        final_energy = E_guess
        P = P_guess
        P_alpha = P_guess / 2
        P_beta = P_guess / 2
        E_MP2 = 0
        E_MP3 = 0

        scf_output = util.Output(final_energy, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals, None, epsilons, epsilons, None)

        if method not in ["HF", "RHF", "UHF"]: util.warning("A correlated calculation has been requested on a one-electron system! Energy will be Hartree-Fock only.")


    else:
    
        if not silent: util.log(" Calculating one and two-electron integrals...  ", calculation, end=""); sys.stdout.flush()
        S, T, V_NE, D, V_EE = integ.evaluate_integrals(atomic_orbitals, np.array(Z_list, dtype=np.float64), coordinates, centre_of_mass)
        if not silent: util.log("[Done]", calculation)

        if not silent: util.log(" Constructing Fock transformation matrix...     ", calculation, end="")
        X = calculate_fock_transformation_matrix(S)
        if not silent: util.log("[Done]", calculation)


        E_guess, P_guess, P_guess_alpha, P_guess_beta, guess_epsilons, guess_mos = setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_electrons_per_orbital, point_group, n_electrons, n_alpha, n_beta, calculation.rotate_guess, calculation.norotate_guess, calculation, silent=silent)


        if not silent: 
            
            print(" Beginning self-consistent field cycle...\n")

            util.log(f" Using \"{calculation.scf_conv.get("word")}\" convergence criteria.", calculation)

            if calculation.diis and not calculation.damping: util.log(" Using DIIS for convergence acceleration.", calculation)
            elif calculation.diis and calculation.damping: util.log(" Using initial dynamic damping and DIIS for convergence acceleration.", calculation)
            elif calculation.damping and not calculation.slowconv and not calculation.veryslowconv: util.log(" Using permanent dynamic damping for convergence acceleration.", calculation)  
            if calculation.slowconv: util.log(" Using strong static damping for convergence acceleration.", calculation)  
            elif calculation.veryslowconv: util.log(" Using very strong static damping for convergence acceleration.", calculation)  
            if calculation.level_shift: util.log(" Using level shift for convergence acceleration.", calculation)
            if not calculation.diis and not calculation.damping and not calculation.level_shift: util.log(" No convergence acceleration used.", calculation)

            util.log("", calculation)


        scf_output, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy = scf.SCF(molecule, calculation, T, V_NE, V_EE, V_NN, S, X, E_guess, P=P_guess, P_alpha=P_guess_alpha, P_beta=P_guess_beta, silent=silent)

        molecular_orbitals = scf_output.molecular_orbitals
        epsilons = scf_output.epsilons
        P = scf_output.P
        P_alpha = scf_output.P_alpha
        P_beta = scf_output.P_beta
        final_energy = scf_output.energy

        scf_output.D = D

        if reference == "UHF": 
                
                if not silent and n_electrons > 1:

                    s_squared, s_squared_exact, spin_contamination = calculate_spin_contamination(P_alpha, P_beta, n_alpha, n_beta, S)

                    util.log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation)
                    util.log("             UHF Spin Contamination       ", calculation)
                    util.log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation)

                    util.log(f"  Exact S^2 expectation value:     {s_squared_exact:.6f}", calculation)
                    util.log(f"  UHF S^2 expectation value:       {s_squared:.6f}", calculation)
                    util.log(f"\n  Spin contamination:              {spin_contamination:.6f}", calculation)

                    util.log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation)

        if not silent: postscf.print_energy_components(nuclear_electron_energy, kinetic_energy, exchange_energy, coulomb_energy, V_NN, calculation)


        if method in ["MP2", "UMP2", "SCS-MP2", "MP3", "UMP3", "SCS-MP3"]: E_MP2, E_MP3, P = calculate_moller_plesset(calculation.mp2_basis, method, molecule, scf_output, V_EE, silent=silent, terse=terse)


    
    if reference == "RHF": P_alpha = P /2; P_beta = P /2

    if not terse and not silent and not calculation.terse: postscf.post_scf_output(molecule, calculation, epsilons, molecular_orbitals, P, S, molecule.ao_ranges, D, P_alpha, P_beta)
    


    if not silent: 
        
        if calculation.reference == "RHF": print("\n Final restricted Hartree-Fock energy: " + f"{final_energy:.10f}")
        if calculation.reference == "UHF": print("\n Final unrestricted Hartree-Fock energy: " + f"{final_energy:.10f}")


    if method in ["MP2", "SCS-MP2", "UMP2"]: 
    
        final_energy += E_MP2
        
        if not silent: print(f" Correlation energy from {method}: " + f"{E_MP2:.10f}\n")
        if not silent: print(" Final single point energy: " + f"{final_energy:.10f}")
    
    elif method in ["MP3", "UMP3", "SCS-MP3"]:
        
        final_energy += E_MP2 + E_MP3

        if not silent:
            if method == "SCS-MP3":

                print(f" Correlation energy from SCS-MP2: " + f"{E_MP2:.10f}")
                print(f" Correlation energy from SCS-MP3: " + f"{E_MP3:.10f}\n")

            else:

                print(f" Correlation energy from MP2: " + f"{E_MP2:.10f}")
                print(f" Correlation energy from MP3: " + f"{E_MP3:.10f}\n")

            print(" Final single point energy: " + f"{final_energy:.10f}")

    if calculation.d2:
    
        final_energy += E_D2

        if not silent: print("\n Semi-empirical dispersion energy: " + f"{E_D2:.10f}")
        if not silent: print(" Dispersion-corrected final energy: " + f"{final_energy:.10f}")
    


    if calculation.densplot and not silent: postscf.construct_electron_density(P, 0.07, molecule)

    final_density = P

    return scf_output, molecule, final_energy, final_density


    

def scan_coordinate(calculation, atoms, starting_coordinates):

    coordinates = util.bohr_to_angstrom(starting_coordinates)
    number_of_steps = calculation.scannumber
    step_size = calculation.scanstep

    print(f"Initialising a {number_of_steps} step coordinate scan in {step_size:.4f} Angstrom increments.") 
    print(f"Starting at a bond length of {np.linalg.norm(coordinates[1] - coordinates[0]):.4f} Angstroms.\n")
    
    bond_lengths = [] ; energies = []   
    P_guess = None; E_guess = None; P_guess_alpha = None; P_guess_beta = None


    for step in range(1, number_of_steps + 1):
        
        bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Starting scan step {step} of {number_of_steps} with bond length of {bond_length:.4f} Angstroms...")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        
        scf_output, molecule, energy, _ = calculate_energy(calculation, atoms, util.angstrom_to_bohr(coordinates), P_guess, P_guess_alpha, P_guess_beta, E_guess, terse=True)

        if calculation.moread: P_guess = scf_output.P; E_guess = energy; P_guess_alpha = scf_output.P_alpha; P_guess_beta = scf_output.P_beta
        else: P_guess = None; E_guess = None


        energies.append(energy)
        bond_lengths.append(bond_length)

        coordinates = np.array([coordinates[0], [0,0,coordinates[1][2] + step_size]])
        
    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    
    print("\nCoordinate scan calculation finished, printing energy values...\n")
    
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("   R (angstroms)    Energy (hartree)")
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
