import __init__ as init
version_number = init.__version__

print("\n      _______ _    _ _   _                     ___           \n     |__   __| |  | | \\ | |   /\\            __/__/__  _      \n ~~~~~~ | |  | |  | |  \\| |  /  \\ ~~~~~~~~ / .      \\/ ) ~~~~\n ~~~~~~ | |  | |  | | . ` | / /\\ \\ ~~~~~~ (     ))    ( ~~~~~\n ~~~~~~ | |  | |__| | |\\  |/ ____ \\ ~~~~~~ \\________/\\_) ~~~~\n        |_|   \\____/|_| \\_/_/    \\_\\          \\\\_\\           ")

print("\n")
print(f"Welcome to version {version_number} of TUNA (Theoretical Unification of Nuclear Arrangements)!\n")
print("Importing required libraries...    ",end="")

import sys; sys.stdout.flush()
import numpy as np
import time
import tuna_basis as basis_sets
import tuna_integral as integ
import tuna_scf as scf
import tuna_postscf as postscf
import tuna_dispersion as disp

print("[Done]\n")

start_time = time.time()

bohr_radius_in_angstrom = 0.52917721090
bohr_radius_in_angstrom_orca = 0.5291772


def bohr_to_angstrom(length): return bohr_radius_in_angstrom * length

def angstrom_to_bohr(length): return length / bohr_radius_in_angstrom 

def one_dimension_to_three(coordinates): return np.array([[0, 0, coord] for coord in coordinates])

def three_dimensions_to_one(coordinates): return np.array([atom_coord[2] for atom_coord in coordinates])
    
def calculate_nuclear_repulsion(Z_list, coordinates): return np.prod(Z_list) / np.linalg.norm(coordinates[1] - coordinates[0])
    
    
def finish_calculation(calculation_type):

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{calculation_type} calculation in TUNA completed successfully in {total_time:.2f} seconds.  :)\n")
    sys.exit()
 

def make_initial_guess(H_Core, X):

    H_Core_orthonormal = X.T@H_Core@X
    guess_epsilons, C_Prime = np.linalg.eigh(H_Core_orthonormal)
    guess_mos = X@C_Prime
    
    P_Guess = scf.construct_density_matrix(guess_mos, n_occ)   
    
    return guess_epsilons, guess_mos, P_Guess


def generate_atomic_orbitals(atom_type, basis, location):
    
    basis = basis.replace("-", "_")
    basis = basis.replace("+", "_plus")
    
    atomic_orbitals = getattr(basis_sets, f"generate_{basis.lower()}_orbitals")(atom_type, location)
    
    return atomic_orbitals
    

def generate_molecule(atoms, coordinates, basis): 
    
    molecule = [generate_atomic_orbitals(atoms[i], basis, coordinates[i]) for i in range(len(atoms))]

    return molecule






def get_params():
    
    input_line = ""
    atom_charges = {"XH": 0, "XHE": 0, "H": 1, "HE": 2}
    atom_masses = {"H": 1837.152708015657, "HE": 7296.2996300303803}
    atom_options = ["XH", "XHE", "H", "HE"]
    calculation_options = ["SPE", "OPT", "SCAN", "FREQ", "OPTFREQ"]
    method_options = ["HF", "RHF", "MP2", "SCS-MP2"]
    basis_options = ["STO-3G", "STO-6G", "3-21G", "6-31G", "6-31+G","6-311+G","6-311G", "6-311++G", "4-31G","HTO-CBS", "6-31++G"]
    Z_list = []; m_list = []

    for arg in range(1, len(sys.argv)): input_line += sys.argv[arg] + " "
    input_line = input_line.upper().strip()
    
    calculation = input_line.split(":")[0].strip()
    
    try: geometry_section = input_line.split(":")[1].split("!")[0].strip() 
    except: sys.exit("ERROR: Input line formatted incorrectly!")

    atoms = [atom.strip() for atom in geometry_section.split(" ")[0:2] if atom.strip()]

    coordinates_1d = [0] + [float(bond_length.strip()) for bond_length in geometry_section.split(" ")[2:] if bond_length.strip()]

    method, basis = input_line.split("!")[1].split("!")[0].strip().split()

    if len(input_line.split("!")) == 3:
        params = input_line.split("!")[2].strip().split()
        for param in params: param = param.strip()
        
    else: params = []
    
    if calculation not in calculation_options: sys.exit(f"ERROR: Calculation type \"{calculation}\" is not supported.  :(")
    if method not in method_options: sys.exit(f"ERROR: Calculation method \"{method}\" is not supported.  :(")
    if basis not in basis_options: sys.exit(f"ERROR: Basis set \"{basis}\" is not supported.  :(")



    for atom in atoms:
        if atom not in atom_options: 
            sys.exit(f"ERROR: Atom type \"{atom}\" not recognised! Available atoms are H, He and ghost atoms XH and XHe.")
        Z_list.append(atom_charges.get(atom))
        m_list.append(atom_masses.get(atom))
    
    if len(atoms) != len(coordinates_1d): sys.exit("ERROR: Two atoms requested without a bond length!  :(")
    
    coordinates = one_dimension_to_three(angstrom_to_bohr(np.array(coordinates_1d)))


    return calculation, method, basis, atoms, Z_list, coordinates, params, m_list


def process_params(params):

    loose = {"delta_E": 0.000001, "maxDP": 0.00001, "rmsDP": 0.000001, "orbitalGrad": 0.0001, "word": "loose"}
    medium = {"delta_E": 0.0000001, "maxDP": 0.000001, "rmsDP": 0.0000001, "orbitalGrad": 0.00001, "word": "medium"}
    tight = {"delta_E": 0.000000001, "maxDP": 0.00000001, "rmsDP": 0.000000001, "orbitalGrad": 0.0000001, "word": "tight"}
    extreme = {"delta_E": 0.00000000001, "maxDP": 0.0000000001, "rmsDP": 0.00000000001, "orbitalGrad": 0.000000001, "word": "extreme"}   
    
    looseopt = {"gradient": 0.01, "step": 0.1, "word": "loose"}
    mediumopt = {"gradient": 0.0001, "step": 0.0001, "word": "medium"}
    tightopt = {"gradient": 0.000001, "step": 0.00001, "word": "tight"}
    extremeopt = {"gradient": 0.00000001, "step": 0.0000001, "word": "extreme"}
    
    charge = 0; max_iter = 30; d2 = False; scf_conv = tight; densplot = False; DIIS = True; geom_max_iter = 30; geom_conv = tightopt; calchess = False; temp = False
    level_shift = False; scanstep = None; scannumber = None; damping = True; additional_print = False; scan_plot=False; slowconv=False; moread = False; pres = False
    veryslowconv = False

    if "CHARGE" in params: 
        try: charge = int(params[params.index("CHARGE") + 1])
        except IndexError: sys.exit("ERROR: Parameter \"CHARGE\" requested but no charge specified!  :(")
        except ValueError: sys.exit("ERROR: Charges must be integers! :(")  
    
    elif "CH" in params: 
        try: charge = int(params[params.index("CH") + 1])
        except IndexError: sys.exit("ERROR: Parameter \"CHARGE\" requested but no charge specified!  :(")
        except ValueError: sys.exit("ERROR: Charges must be integers! :(")  

    if "MAXITER" in params: 
        try: max_iter = int(params[params.index("MAXITER") + 1])
        except IndexError: sys.exit("ERROR: Parameter \"MAXITER\" requested but no maximum SCF iterations specified!  :(")
        except ValueError: sys.exit("ERROR: Maximum SCF iterations must be an integer!  :(") 

    if "GEOMMAXITER" in params: 
        try: geom_max_iter = int(params[params.index("GEOMMAXITER") + 1])
        except IndexError: sys.exit("ERROR: Parameter \"GEOMMAXITER\" requested but no maximum geometry iterations specified!  :(")
        except ValueError: sys.exit("ERROR: Maximum geometry iterations must be an integer!  :(")         

    if "MAXGEOMITER" in params: 
        try: geom_max_iter = int(params[params.index("MAXGEOMITER") + 1])
        except IndexError: sys.exit("ERROR: Parameter \"MAXGEOMITER\" requested but no maximum geometry iterations specified!  :(")
        except ValueError: sys.exit("ERROR: Maximum geometry iterations must be an integer!  :(")      
    
    if "SCANSTEP" in params: 
        try: scanstep = float(params[params.index("SCANSTEP") + 1])
        except IndexError: sys.exit("ERROR: Parameter \"SCANSTEP\" requested but no step length for scan calculation specified!  :(")
        except ValueError: sys.exit("ERROR: Coordinate scan step size must be a floating point number!  :(")
        
    if "SCANNUMBER" in params: 
        try: scannumber = int(params[params.index("SCANNUMBER") + 1])
        except IndexError: sys.exit("ERROR: Parameter \"SCANNUMBER\" requested but no number of scan steps specified!  :(")
        except ValueError: sys.exit("ERROR: Number of coordinate scan steps must be an integer!  :(") 
        
    if "TEMP" in params: 
        try: temp = int(params[params.index("TEMP") + 1])
        except IndexError: sys.exit("ERROR: Parameter \"TEMP\" requested but no temperature specified!  :(")
        except ValueError: sys.exit("ERROR: Temperature must be a floating point number!  :(") 
    
    if "PRES" in params: 
        try: pres = int(params[params.index("PRES") + 1])
        except IndexError: sys.exit("ERROR: Parameter \"PRES\" requested but no pressure specified!  :(")
        except ValueError: sys.exit("ERROR: Pressure must be a floating point number!  :(") 
        
    if "LOOSE" in params or "LOOSESCF" in params: scf_conv = loose  
    elif "MEDIUM" in params or "MEDIUMSCF" in params: scf_conv = medium  
    elif "TIGHT" in params or "TIGHT" in params: scf_conv = tight   
    elif "EXTREME" in params or "EXTREMESCF" in params: scf_conv = extreme 
    
    if "LOOSEOPT" in params: geom_conv = looseopt  
    elif "MEDIUMOPT" in params: geom_conv = mediumopt  
    elif "TIGHTOPT" in params: geom_conv = tightopt 
    elif "EXTREMEOPT" in params: geom_conv = extremeopt    
    
    if "SLOWCONV" in params: 
        damping = True
        slowconv = True

    elif "VERYSLOWCONV" in params: 
        damping = True
        veryslowconv = True

    if "NODIIS" in params: DIIS = False 
    if "DAMP" in params: damping = True
    elif "NODAMP" in params: damping = False
    if "DENSPLOT" in params: densplot = True
    if "LEVELSHIFT" in params: level_shift = True
    if "SCANPLOT" in params: scan_plot = True
    
    if "D2" in params: d2 = True
    if "CALCHESS" in params: calchess = True
    if "P" in params: additional_print = True
    if "MOREAD" in params: moread = True

    if moread: scf_conv = extreme

    return charge, max_iter, d2, scf_conv, densplot, DIIS, level_shift, scanstep, scannumber, damping, additional_print, scan_plot, geom_max_iter, geom_conv, slowconv, calchess, moread, temp, pres, veryslowconv
    

def calculate_fock_transformation_matrix(S):
        
    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    
    X = np.linalg.inv(S_sqrt)

    return X


def calculate_energy(coordinates, P_Guess=0, guess_energy=0,terse=False):

    print("\n Setting up molecule...  ",end=""); sys.stdout.flush()
    molecule = generate_molecule(atoms, coordinates, basis)
    orbitals = [orbital for atom in molecule for orbital in atom]
    pgs = [pg for orbital in orbitals for pg in orbital]

    if len(atoms) == 2: molecular_structure = atoms[0].lower().capitalize() + " --- " + atoms[1].lower().capitalize()
    else: molecular_structure = atoms[0].lower().capitalize()
    
       
    print("[Done]\n")
        
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("  Molecular structure: " + molecular_structure)
    print("  Number of atoms: " + str(len(atoms)))
    print("  Number of basis functions: " + str(len(orbitals)))
    print("  Number of primitive Gaussians: " + str(len(pgs)))
    print("  Charge: " + str(charge))
    print("  Number of electrons: " + str(n_electrons))

    if len(atoms) == 2: 
        if atoms[0] == atoms[1]: point_group = "Dinfh"
        else: point_group = "Cinfv"
        bond_length = bohr_to_angstrom(np.linalg.norm(coordinates[1] - coordinates[0]))
        print(f"  Bond length: {bond_length:.4f} ")
        print(f"  Point group: {point_group}")
    else: point_group = "K"
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    if len(Z_list) == 2 and "X" not in atoms:

        print(" Calculating nuclear repulsion energy...  ",end="")
        V_NN = calculate_nuclear_repulsion(Z_list, coordinates)
        print("[Done]")
        print(f" Nuclear repulsion energy: {V_NN:.9f}\n")
        
        if d2:     
                print(" Calculating semi-empirical dispersion energy...  ",end="")
                E_D2 = disp.calculate_d2_energy(atoms, coordinates)
                print("[Done]")
                print(f" Dispersion energy (D2): {E_D2:.9f}\n")
            
        else: E_D2 = 0
        
    else: V_NN = 0; E_D2 = 0
        
        



    if n_electrons > 0:
        
        if n_electrons % 2 != 0 and n_electrons != 1:
            sys.exit("ERROR: Only restricted Hartree-Fock is implemented. Multi-electron systems must be closed shell!  :(")
            
    elif n_electrons == 0: 
    
        guess_energy = 0
        print("WARNING: Calculation specified with zero electrons!\n")
        print("Final energy: 0.000000000\n")
        
        finish_calculation(calculation_types.get(calculation))
        
    else: sys.exit(f"ERROR: Negative number of electrons ({n_electrons}) specified!  :(")


    if n_electrons > 1:
    
        print(" Calculating one and two-electron integrals...  ",end=""); sys.stdout.flush()
        S, T, V_NE, V_EE = integ.calculate_integrals(orbitals, Z_list, coordinates)
        print("[Done]")

        print(" Constructing Fock transformation matrix...     ",end="")
        X = calculate_fock_transformation_matrix(S)
        print("[Done]")


        if type(P_Guess) != np.ndarray:
            print(" Calculating one-electron density for guess...  ",end="")
            H_Core = T + V_NE
            guess_epsilons, _, P_Guess = make_initial_guess(H_Core, X)
            guess_energy = guess_epsilons[0]
            print("[Done]\n")
        else:
            print("\n Using density matrix from previous step for guess... \n")

        print(" Beginning self-consistent field cycle...\n")

        print(f" Using \"{scf_conv.get("word")}\" convergence criteria.")
        if DIIS and not damping: print(" Using DIIS for convergence acceleration.")
        elif DIIS and damping: print(" Using initial dynamic damping and DIIS for convergence acceleration.")
        elif damping and not slowconv and not veryslowconv: print(" Using permanent dynamic damping for convergence acceleration.")  
        if slowconv: print(" Using strong static damping for convergence acceleration.")  
        elif veryslowconv: print(" Using very strong static damping for convergence acceleration.")  
        if level_shift: print(" Using level shift for convergence acceleration.")
        if not DIIS and not damping and not level_shift: print(" No convergence acceleration used.")
        print("")
        final_energy, molecular_orbitals, epsilons, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy, P = scf.SCF(X, T, V_NE, V_EE, P_Guess, guess_energy, S, max_iter, n_electrons, V_NN, scf_conv, DIIS, level_shift,damping,slowconv, veryslowconv)
        
        postscf.print_energy_components(nuclear_electron_energy, kinetic_energy, exchange_energy, coulomb_energy, V_NN)




        if method == "MP2" or method == "SCS-MP2":
            print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("       MP2 Energy and Density Calculation ")
            print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            import tuna_mp2 as mp2
            
            P_HF_mo = molecular_orbitals.T @ S @ P @ S @ molecular_orbitals

            occupied_mos = molecular_orbitals[:, :n_occ]
            virtual_mos = molecular_orbitals[:, n_occ:]
            
            occupied_epsilons = epsilons[:n_occ]
            virtual_epsilons = epsilons[n_occ:]

            V_EE_mo = mp2.transform_ao_two_electron_integrals(V_EE, occupied_mos, virtual_mos)
   

            if method == "MP2": E_MP2, P_MP2_MO = mp2.calculate_mp2_energy_and_density(occupied_epsilons, virtual_epsilons, V_EE_mo, P_HF_mo, silent=False)
            elif method == "SCS-MP2": E_MP2, E_MP2_OS, E_MP2_SS, P_MP2_MO = mp2.calculate_scs_mp2_energy_and_density(occupied_epsilons, virtual_epsilons, V_EE_mo, P_HF_mo, silent=False)
        
            P = molecular_orbitals @ P_MP2_MO @ molecular_orbitals.T

            natural_orbital_occupancies = np.sort(np.linalg.eigh(P_MP2_MO)[0])[::-1]
            sum_of_occupancies = np.sum(natural_orbital_occupancies)

            print("\n  Natural orbital occupancies: \n")

            for i, natural_orbital in enumerate(natural_orbital_occupancies):

                print(f"    {i + 1}.   {natural_orbital_occupancies[i]:.8f}")
    
            print(f"\n  Sum of natural orbital occupancies: {sum_of_occupancies:.6f}")
            print(f"  Trace of density matrix:  {np.trace(P_MP2_MO):.6f}")
    
                    
            print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
        else: E_MP2 = 0
        
        
    else: 

        print(" Calculating one-electron integrals...    ",end=""); sys.stdout.flush()
        S, T, V_NE = integ.calculate_one_electron_integrals(orbitals, Z_list, coordinates)
        print("[Done]")     

        print(" Constructing Fock transformation matrix...  ",end="")
        X = calculate_fock_transformation_matrix(S)
        print("[Done]")


        print(" Calculating one-electron energy...  ",end="")
        H_Core = T + V_NE
        epsilons, molecular_orbitals, P = make_initial_guess(H_Core, X)
        print("[Done]\n")

        final_energy = epsilons[0] + V_NN
        E_MP2 = 0

        if calculation == "MP2" or calculation == "SCS-MP2":
            print("WARNING: An MP2 calculation has been requested on a one-electron system! Energy will be Hartree-Fock only.")

    ao_ranges = [len(generate_atomic_orbitals(atom, basis, coord)) for atom, coord in zip(atoms, coordinates)]
    
    if not terse: postscf.post_scf_output(method, additional_print, n_electrons, epsilons, molecular_orbitals, n_occ, m_list, Z_list, coordinates, molecular_structure, molecule, atoms, P, S, ao_ranges, orbitals)
    
    
    print("\n Final Hartree-Fock energy: " + f"{final_energy:.9f}")
    if d2:
    
        final_energy += E_D2
        
        print(" Dispersion-corrected final energy: " + f"{final_energy:.9f}")
    
    if method == "MP2" or method == "SCS-MP2": 
    
        final_energy += E_MP2
        
        print(f" Correlation energy from {method}: " + f"{E_MP2:.9f}\n")
        print(" Final single point energy: " + f"{final_energy:.9f}")
    

    if densplot: n = postscf.construct_electron_density(P, 0.07, coordinates, orbitals, n_occ)



    return final_energy, epsilons, molecular_orbitals, P, S, ao_ranges, molecule, molecular_structure, orbitals, point_group


def calculate_silent_energy(coordinates): 

    molecule = generate_molecule(atoms, coordinates, basis)
    orbitals = [orbital for atom in molecule for orbital in atom]


    if len(Z_list) == 2: 
        
        V_NN = calculate_nuclear_repulsion(Z_list, coordinates)
    
        if d2: E_D2 = disp.calculate_d2_energy(atoms, coordinates)     
        else: E_D2 = 0
        
    else: V_NN = 0; E_D2 = 0
        

    if n_electrons > 1:
    
        S, T, V_NE, V_EE = integ.calculate_integrals(orbitals, Z_list, coordinates)
        X = calculate_fock_transformation_matrix(S)
        H_Core = T + V_NE


        guess_epsilons, _, P_Guess = make_initial_guess(H_Core, X)
        
        final_energy, molecular_orbitals, epsilons, _, _, _, _, P, = scf.SCF(X, T, V_NE, V_EE, P_Guess, guess_epsilons[0], S, max_iter, n_electrons, V_NN, scf_conv, DIIS, level_shift,damping,slowconv, veryslowconv, silent=True)


        if method == "MP2" or method == "SCS-MP2":

            import tuna_mp2 as mp2
            P_HF_mo = molecular_orbitals.T @ S @ P @ S @ molecular_orbitals

            occupied_mos = molecular_orbitals[:, :n_occ]
            virtual_mos = molecular_orbitals[:, n_occ:]
            
            occupied_epsilons = epsilons[:n_occ]
            virtual_epsilons = epsilons[n_occ:]

            V_EE_mo = mp2.transform_ao_two_electron_integrals(V_EE, occupied_mos, virtual_mos, silent=True)

            if method == "MP2": E_MP2, P_MP2_MO = mp2.calculate_mp2_energy_and_density(occupied_epsilons, virtual_epsilons, V_EE_mo, P_HF_mo, silent=True)
            elif method == "SCS-MP2": E_MP2, E_MP2_OS, E_MP2_SS, P_MP2_MO = mp2.calculate_scs_mp2_energy_and_density(occupied_epsilons, virtual_epsilons, V_EE_mo, P_HF_mo, silent=True)
                

        else: E_MP2 = 0
        
        
    elif n_electrons == 1: 

        S, T, V_NE = integ.calculate_one_electron_integrals(orbitals, Z_list, coordinates)
        X = calculate_fock_transformation_matrix(S)
        
        H_Core = T + V_NE
        epsilons, _, P = make_initial_guess(H_Core, X)
        
        final_energy = epsilons[0] + V_NN
        E_MP2 = 0



    final_energy += E_D2 + E_MP2

    return final_energy, P, orbitals
    

def scan_coordinate(starting_coordinates, step_size, number_of_steps):

    bond_lengths = []
    energies = []
    coordinates = starting_coordinates

    print(f"Initialising a {number_of_steps} step coordinate scan in {bohr_to_angstrom(step_size):.4f} Angstrom increments.") 
    print(f"Starting at a bond length of {bohr_to_angstrom(np.linalg.norm(starting_coordinates[1] - starting_coordinates[0])):.4f} Angstroms.\n")
    
    P_guess = 0
    energy_guess = 0

    for step in range(1, number_of_steps + 1):
        
        bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Starting scan step {step} of {number_of_steps} with bond length of {bohr_to_angstrom(bond_length):.4f} Angstroms...")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        
        if not moread:
            P_guess = 0
            energy_guess = 0

        energy, epsilons, molecular_orbitals, P, S, ao_ranges, molecule, molecular_structure, orbitals, point_group = calculate_energy(coordinates, P_guess, energy_guess, terse=True)

        P_guess = P
        energy_guess = energy


        energies.append(energy)
        bond_lengths.append(bohr_to_angstrom(bond_length))

        coordinates = np.array([coordinates[0], [0,0,coordinates[1][2] + step_size]])
        
    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    
    print("\nCoordinate scan calculation finished, printing energy values...\n")
    
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("   R (Angstroms)    Energy (Hartree)")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for energy, bond_length in zip(energies,bond_lengths):
        if energy > 0: energy_f = " " + f"{energy:.9f}"
        else: energy_f = f"{energy:.9f}"
        print(f"      {bond_length:.4f}          {energy_f}")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    
    import tuna_anharm as anharm
    print(219474*anharm.calculate_anharmonic_frequency(energies, angstrom_to_bohr(np.array(bond_lengths))))

    if scan_plot:
    
        print("Plotting energy profile diagram...   ",end=""); sys.stdout.flush()
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(10,5))    
        plt.plot(bond_lengths, energies, color=(0.75,0,0),linewidth=1.75)
        plt.xlabel("Bond Length (Angstrom)", fontweight="bold", labelpad=10, fontfamily="Arial",fontsize=12)
        plt.ylabel("Energy (Hartree)",labelpad=10, fontweight="bold", fontfamily="Arial",fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.25, length=6, direction='out')
        ax.tick_params(axis='both', which='minor', labelsize=11, width=1.25, length=3, direction='out')
        
        for spine in ax.spines.values(): spine.set_linewidth(1.25)
        
        plt.minorticks_on()
        plt.tight_layout() 
        print("[Done]")
        
        
        plt.show()


def calculate_gradient(coordinates, energy):

    prod = 0.0001

    forward_coords = np.array([[0, 0, 0], [0, 0, coordinates[1][2] + prod]])
    energy_prod_plus, _, _ = calculate_silent_energy(forward_coords)

    deltaE = energy_prod_plus - energy

    gradient = deltaE / prod

    return gradient
    

def calculate_approximate_hessian(delta_x, delta_grad): 

    hessian = delta_grad / delta_x

    return hessian


def calculate_hessian(energy, coordinates):
    
    prod = 0.0001

    forward_coords = np.array([[0,0,0],[0,0,coordinates[1][2] + prod]])  
    backwards_coords = np.array([[0,0,0],[0,0,coordinates[1][2] - prod]])  
    
    forward_energy,_,_ = calculate_silent_energy(forward_coords)
    backward_energy,_,_ = calculate_silent_energy(backwards_coords)

    hessian = (forward_energy - 2 * energy + backward_energy) / prod ** 2

    return hessian



def optimise_geometry(starting_coordinates, max_geom_iterations, geom_conv_criteria):
    
    maximum_step = angstrom_to_bohr(0.2)

    default_hess = 1/4
    P_guess = 0
    energy_guess = 0

    coordinates = starting_coordinates
    print("\nInitialising geometry optimisation...\n")
    if not calchess: print(f"Using approximate Hessian in convex region, Hessian of {default_hess:.3f} outside.")
    else: print(f"Using exact Hessian in convex region, Hessian of {default_hess:.3f} outside.")
    print(f"Gradient convergence: {geom_conv_criteria.get("gradient"):.7f}")
    print(f"Step convergence: {geom_conv_criteria.get("step"):.7f}")
    print(f"Maximum iterations: {max_geom_iterations}")
    print(f"Maximum step: {bohr_to_angstrom(maximum_step):.5f}")

    for iteration in range(1, max_geom_iterations + 1):
        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Beginning energy and gradient calculation on geometry iteration number {iteration}...")
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        
        if not moread: 
            P_guess = 0
            energy_guess = 0

        energy, epsilons, molecular_orbitals, P, S, ao_ranges, molecule, molecular_structure, orbitals, point_group = calculate_energy(coordinates, P_guess, energy_guess, terse=True)
        
        P_guess = P
        energy_guess = energy

        print("\n Calculating numerical gradient...  ", end=""); sys.stdout.flush()
        gradient = calculate_gradient(coordinates, energy)
        print("[Done]")
        bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

        if gradient > 0: space = "  "
        else: space = "  "
     
        if iteration > 1:

            hessian = 1/4
            if calchess: h = calculate_hessian(energy, coordinates)
            else: h = calculate_approximate_hessian(bond_length-old_bond_length, gradient-old_gradient)

            if h > 0.01: hessian = h

        else: hessian = 1 / 4

        inverse_hessian = 1 / hessian
           
        step = inverse_hessian * gradient
        

        if np.abs(gradient) < geom_conv_criteria.get("gradient"): converged_grad = True; conv_check_grad = "Yes"
        else: converged_grad = False; conv_check_grad = "No"

        if np.abs(step) < geom_conv_criteria.get("step"): converged_step = True; conv_check_step = "Yes"
        else: converged_step = False; conv_check_step = "No"
        
        print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("   Factor       Value     Conv. Criteria   Converged")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"  Gradient    {gradient:.7f}  {space} {geom_conv_criteria.get("gradient"):.7f}   {space}    {conv_check_grad} ")
        print(f"    Step      {step:.7f}  {space} {geom_conv_criteria.get("step"):.7f}   {space}    {conv_check_step} ")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


        if converged_grad and converged_step: 
            print("\n=========================================")           
            print(f" Optimisation converged in {iteration} iterations!")
            print("=========================================")
            postscf.post_scf_output(method, additional_print, n_electrons, epsilons, molecular_orbitals, n_occ, m_list, Z_list, coordinates, molecular_structure, molecule, atoms, P, S, ao_ranges, orbitals)
            
            
            print(f"\n Optimisation converged in {iteration} iterations to bond length of {bohr_to_angstrom(bond_length):.4f} Angstroms!")
            print(f"\n Final single point energy: {energy:.9f}")

            return coordinates, energy, P, orbitals, point_group

        else:
            
            if step > maximum_step: 
                step = maximum_step
                print("WARNING: Calculated step is outside of trust radius, taking maximum step instead.")
            if step < -maximum_step:
                step = -maximum_step
                print("WARNING: Calculated step is outside of trust radius, taking maximum step instead.")

            coordinates = np.array([[0, 0, 0], [0, 0, coordinates[1][2] - step]])

            if coordinates[1][2] <= 0: sys.exit("ERROR: Optimisation generated negative bond length! Decrease trust radius!  :()")

            old_bond_length = bond_length
            old_gradient = gradient
     

    sys.exit(f"\nWARNING: Geometry optimisation did not converge in {max_geom_iterations} iterations! Increase the maximum or give up!")


def calculate_frequency(coordinates, optimised_energy=0, point_group_from_opt=0):
    if calculation != "OPTFREQ":
          energy, epsilons, molecular_orbitals, P, S, ao_ranges, molecule, molecular_structure, orbitals, point_group = calculate_energy(coordinates)
    else:
        energy = optimised_energy
        point_group = point_group_from_opt

    bond_length = np.abs(coordinates[1][2] - coordinates[0][2])

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Beginning TUNA harmonic frequency calculation...")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    
    print(f"\n Calculating numerical Hessian at bond length of {bohr_to_angstrom(bond_length):.4f} Angstroms...  ", end=""); sys.stdout.flush()
    
    
    k = calculate_hessian(energy, coordinates)
    reduced_mass = postscf.calculate_reduced_mass(m_list)
    if k > 0:
        frequency_hartree = np.sqrt(k / reduced_mass)
        i = ""
        imaginary_freq = False
    else:   
        frequency_hartree = np.sqrt(-k / reduced_mass)
        i = " i"
        imaginary_freq = True

    frequency_per_cm = frequency_hartree * 219474.63068

    print("[Done]\n")


    print(" Using masses of most abundant isotopes...\n")

    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("       Harmonic Frequency")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"  Force constant: {k:.5f}")
    print(f"  Reduced mass: {reduced_mass:.2f}")
    print(f"\n  Frequency (per cm): {frequency_per_cm:.2f}{ i}")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    zpe = 0.5 * frequency_hartree

    import tuna_thermo as thermo

    if temp == False: temperature = 298.15
    else: temperature = temp

    if pres == False: pressure = 101325
    else: pressure = pres

    if imaginary_freq: zpe = 0

    print(f"\n Temperature used is {temperature:.2f} K, pressure used is {(pressure)} Pa.")
    print(" Entropies multiplied by temperature to give units of energy.")
    print(f" Using symmetry number derived from {point_group} point group for rotational entropy.")

    rotational_constant_per_cm, rotational_constant_GHz = postscf.calculate_rotational_constant(m_list, coordinates)

    U, translational_internal_energy, rotational_internal_energy, vibrational_internal_energy = thermo.calculate_internal_energy(energy, zpe, temperature, frequency_per_cm)
    H = thermo.calculate_enthalpy(U, temperature)

    S, translational_entropy, rotational_entropy, vibrational_entropy, electronic_entropy = thermo.calculate_entropy(temperature, frequency_per_cm, point_group, rotational_constant_per_cm * 100, m_list, coordinates, pressure)
    G = H - temperature * S

    if imaginary_freq: vibrational_entropy = 0; vibrational_internal_energy = 0

    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("                                 Thermochemistry")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"  Electronic energy:    {energy:.9f}       Electronic entropy:      {temperature*electronic_entropy:.9f}")
    print(f"\n  Zero-point energy:     {zpe:.9f}")
    print(f"  Translational energy:  {translational_internal_energy:.9f}       Translational entropy:   {temperature*translational_entropy:.9f}")
    print(f"  Rotational energy:     {rotational_internal_energy:.9f}       Rotational entropy:      {temperature*rotational_entropy:.9f}")
    print(f"  Vibrational energy:    {vibrational_internal_energy:.9f}       Vibrational entropy:     {temperature*vibrational_entropy:.9f}  ")
    print(f"\n  Internal energy:      {U:.9f}       Entropy:                 {temperature*S:.9f}")
    print(f"  Enthalpy:             {H:.9f}")
    print(f"\n  Gibbs free energy:    {G:.9f}       Non-electronic energy:   {energy - G:.9f}")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def main(): 


    
    global calculation; global method; global basis; global atoms; global Z_list; global m_list; global geom_conv; global slowconv; global calchess; global moread; global calculation_types
    global charge; global max_iter; global d2; global scf_conv; global densplot; global DIIS; global level_shift; global scanstep; global scannumber; global damping; global additional_print; global scan_plot
    global temp; global pres; global veryslowconv

    calculation, method, basis, atoms, Z_list, coordinates, params, m_list = get_params()

    calculation_types = {"SPE": "Single point energy", "OPT": "Geometry optimisation", "FREQ": "Harmonic frequency", "OPTFREQ": "Optimisation and harmonic frequency", "SCAN": "Coordinate scan"}
    method_types = {"HF": "restricted Hartree-Fock theory", "RHF": "restricted Hartree-Fock theory", "MP2": "MP2 theory", "SCS-MP2": "spin-component scaled MP2 theory"}

    print(f"{calculation_types.get(calculation)} calculation in \"{basis}\" basis set via {method_types.get(method)} requested.")
    print(f"Everything except distances in atomic units. Distances in angstroms. ")


    charge, max_iter, d2, scf_conv, densplot, DIIS, level_shift, scanstep, scannumber, damping, additional_print, scan_plot, geom_max_iter, geom_conv, slowconv, calchess, moread, temp, pres, veryslowconv = process_params(params)
    
    global n_electrons; global n_occ
    
    n_electrons = np.sum(Z_list) - charge
    n_occ = int(n_electrons / 2)


    if calculation == "SPE": 

        calculate_energy(coordinates)
        finish_calculation(calculation_types.get(calculation))
        

    elif calculation == "SCAN":

        if scanstep:
            if scannumber: 
            
                scan_coordinate(coordinates, angstrom_to_bohr(scanstep), scannumber)
                finish_calculation(calculation_types.get(calculation))
                
            else: sys.exit("ERROR: Coordinate scan requested but no number of steps given by keyword \"SCANNUMBER\"!  :(" )
        else: sys.exit("ERROR: Coordinate scan requested but no step size given by keyword \"SCANSTEP\"!  :(" )
        

    elif calculation == "OPT":
        if len(atoms) == 1: sys.exit("\nERROR: Geometry optimisation requested for single atom!")
        optimise_geometry(coordinates, geom_max_iter, geom_conv)
        finish_calculation(calculation_types.get(calculation))


    elif calculation == "FREQ":
        if len(atoms) == 1: sys.exit("\nERROR: Harmonic frequencies requested for single atom!")
        calculate_frequency(coordinates)
        finish_calculation(calculation_types.get(calculation))

    elif calculation == "OPTFREQ":
        if len(atoms) == 1: sys.exit("\nERROR: Geometry optimisation requested for single atom!")
        optimised_coordinates, optimised_energy, P, orbitals, point_group = optimise_geometry(coordinates, geom_max_iter, geom_conv)
        calculate_frequency(optimised_coordinates, optimised_energy, point_group)
        finish_calculation(calculation_types.get(calculation))
        
        
""" Features in TUNA and TUNA Roadmap """

#Version 0.1.0

#Basis sets: STO-3G, STO-6G, 6-31G, 6-311G, 6-311++G, HTO-CBS
#Methods: RHF
#Calculation types: Single point energy, coordinate scan
#SCF Convergence: Dynamic damping, level shift
#Post-SCF: Molecular orbitals, orbital energies, Koopman's theorem parameters, HOMO-LUMO gap, electron density plot
#Dispersion correction: D2
#Miscellaneous: loose, medium, tight, extreme convergence criteria for SCF, electron density plot, coordinate scan plot, ghost atoms

#Version 0.2.0
#Methods: MP2, SCS-MP2 energies
#Post-SCF: Mulliken charges and bond order, Lowdin charges and bond order
#Keywords: Additional print, damping options, scan plot option
#Miscellaneous: Python 3.12, density plots for coordinate scan, redesigned logo, increased code efficiency, 
    #fixed coordinate scan working incorrectly, identification of point group, fixed angstrom-bohr conversion factor

#Version 0.3.0
#Calculation types: Geometry optimisation, harmonic frequency, optimisation+frequency
#SCF Convergence: Slow convergence with high static damping, significantly improved integral engine (1.5 - 20x faster)
#Post-SCF: Rotational constants and center of mass, nuclear dipole moment
#Miscellaneous: improved efficiency of MP2 module, optional exact or approximate Hessian for optimisation in convex region, 
    #geometry optimisation and scan are sensible about when to print things,
        #fixed SCS-MP2 not working, geometry convergence keywords and max iterations, made scan plot default off, trust radius for optimisation


#Version 0.4.0
#Basis sets: 3-21G
#SCF: DIIS, unbroke level shift
#Post-SCF: Electronic dipole moment, MP2 unrelaxed density, properties and natural orbital occupancies, thermochemistry in frequency calculation
#Miscellaneous: read MOs from previous point for scan coordinate and optimisation, temperature keyword for thermochemistry and pressure

#For Version 0.5.0
#Calculation types: harmonic frequency intensities, anharmonic frequency, MD
#Methods: UHF, UMP2
#SCF: Improve integral engine, decontract basis keyword
#Basis sets: Formally added 6-31+G and 6-311+G, 4-31G, 6-31++G
#Post-SCF: Mayer bond order, MD runs and geom opts (optionally) print to xyz trajectory file
#Miscellaneous: Slimmed down fish logo :( 



#For Version 0.6.0
#Methods: MP3, SCS-MP3

#For Version 0.7.0
#Methods: CIS, Polarisability

#For Version 0.8.0
#Dispersion: D3
#Methods: CISD, CCSD

#For Version 0.9.0
#Methods: CCSD(T), MP4

#For Version 1.0.0
#Calculation types: Paramagnetic shielding
#Miscellaneous: Rewrite integral engine with p, d and f functions

#For Version 2.0.0
#Methods: DFT with LDA, GGA, hybrid and double-hybrid functionals
#Dispersion: D4

#For Version 3.0.0
#Methods: relativistic corrections, exact-2-component

#For Version 4.0.0
#Calculation types: Analytical first and second derivatives 






if __name__ == "__main__": main()


#Potential ORCA bugs
#too quickly rounded bohr definition
#SCS-MP2 frequency different
#Masses different