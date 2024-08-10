import numpy as np
import tuna_integral as integ
import sys

c = 137.035999

def calculate_centre_of_mass(masses, coordinates): return np.einsum("i,ij->", masses, coordinates) / np.sum(masses)
   
def calculate_nuclear_dipole_moment(centre_of_mass, Z_list, coordinates): 

    nuclear_dipole_moment = 0
    
    for i in range(len(Z_list)):
        nuclear_dipole_moment += (coordinates[i][2] - centre_of_mass) * Z_list[i]
        
    return nuclear_dipole_moment
   
   
def calculate_electronic_dipole_moment(P, D): return -np.einsum("ij,ij->",P,D)

    
def calculate_reduced_mass(masses): return np.prod(masses) / np.sum(masses)


def calculate_rotational_constant(masses, coordinates):

    bond_length = np.linalg.norm(coordinates[1] - coordinates[0])
    reduced_mass = calculate_reduced_mass(masses)
    
    rotational_constant_hartree = 1 / (2 * reduced_mass * bond_length ** 2)  
    rotational_constant_per_bohr = rotational_constant_hartree / (2 * np.pi * c)
    
    rotational_constant_per_cm = 0.01 * rotational_constant_per_bohr / 5.2917721067121e-11 
    rotational_constant_GHz = 29.9792458 * rotational_constant_per_cm
    
    return rotational_constant_per_cm, rotational_constant_GHz


def population_analysis(P, S, coordinates, Z_list, atoms, ao_ranges):
    
    
    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    
    P_reshaped = P.reshape(len(atoms), -1, P.shape[1])
    S_reshaped = S.reshape(S.shape[0], len(atoms), -1)

    qs_mull = Z_list - np.sum(np.multiply(S_reshaped.transpose(1, 2, 0), P_reshaped), axis=(1, 2))
    qsum_mull = np.sum(qs_mull)
    bo_mull = 2 * np.sum(P[:ao_ranges[0], ao_ranges[0]:ao_ranges[0] + ao_ranges[1]] * S[:ao_ranges[0], ao_ranges[0]:ao_ranges[0] + ao_ranges[1]])
    
    
    P_low = S_sqrt @ P @ S_sqrt
    
    qs_low = Z_list - np.sum(np.diag(P_low).reshape(len(atoms), -1)[:, :ao_ranges[0] + ao_ranges[1]], axis=1)    
    qsum_low = np.sum(qs_low)
    bo_low = np.sum(P_low[:ao_ranges[0], ao_ranges[0]:ao_ranges[0] + ao_ranges[1]] ** 2)     
    
    return qs_mull, qsum_mull, bo_mull, qs_low, qsum_low, bo_low
    

def calculate_koopman_parameters(epsilons, n_occ):

    ionisation_energy = -1 * epsilons[n_occ - 1]
        
    if len(epsilons) > n_occ: 
    
        electron_affinity = -1 * epsilons[n_occ]
        homo_lumo_gap = ionisation_energy - electron_affinity
        
    else: 
    
        electron_affinity = "---"
        homo_lumo_gap = "---"
        
        print("WARNING: Size of basis is too small for electron affinity calculation!")


    return ionisation_energy, electron_affinity, homo_lumo_gap
 
 
 
def construct_electron_density(P, grid_density, coords, orbitals, n_occ):

    print("\nBeginning electron density surface plot calculation...\n")

    print("Setting up grid...   ", end="")
    
    coordinates = [coords[0][2], coords[1][2]]
    start = coordinates[0] - 4
    
    x = np.arange(start, coordinates[0] + 4 + grid_density, grid_density)
    y = np.arange(start, coordinates[0] + 4 + grid_density, grid_density)
    z = np.arange(start, coordinates[1] + 4 + grid_density, grid_density)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    
    print("[Done]")
    
    print("Generating electron density cube...   ", end=""); sys.stdout.flush()
    
    n = 0


    atomic_orbitals = []

    for orbital in orbitals:
        a = 0
        for pg in orbital:  
        
            a += pg.N * pg.coeff * np.exp(-pg.alpha * ((X - pg.coordinates[0])**2 + (Y - pg.coordinates[1])**2 + (Z - pg.coordinates[2])**2))
        
        
        atomic_orbitals.append(a)
    
    atomic_orbitals = np.array(atomic_orbitals)
    
    n = np.einsum("mn,mijk,nijk->ijk", P, atomic_orbitals, atomic_orbitals)
    
    normalisation = np.trapz(np.trapz(np.trapz(n,z),y), x)
    n *= n_occ * 2 / normalisation

    print("[Done]")
    print("Generating surface plot...   ", end="")
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


def print_energy_components(nuclear_electron_energy, kinetic_energy, exchange_energy, coulomb_energy, V_NN):

    one_electron_energy = nuclear_electron_energy + kinetic_energy
    two_electron_energy = exchange_energy + coulomb_energy
    electronic_energy = one_electron_energy + two_electron_energy
    total_energy = electronic_energy + V_NN
            
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")      
    print("              Energy Components       ")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            

    print(f"  Kinetic energy:              {kinetic_energy:.9f}")

    print(f"  Coulomb energy:              {coulomb_energy:.9f}")
    print(f"  Exchange energy:            {exchange_energy:.9f}")
    print(f"  Nuclear repulsion energy:    {V_NN:.9f}")
    print(f"  Nuclear attraction energy:  {nuclear_electron_energy:.9f}\n")      

    print(f"  One-electron energy:        {one_electron_energy:.9f}")
    print(f"  Two-electron energy:         {two_electron_energy:.9f}")
    print(f"  Electronic energy:          {electronic_energy:.9f}\n")
            
    print(f"  Total energy:               {total_energy:.9f}")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")  

    return


def post_scf_output(method, additional_print, n_electrons, epsilons, molecular_orbitals, n_occ, m_list, Z_list, coordinates, molecular_structure, molecule, atoms, P, S, ao_ranges, orbitals):

    print("\n Beginning calculation of TUNA properties... ")
        

    if method == "MP2": print("\n Using the MP2 unrelaxed density for property calculations.")
    if method == "SCS-MP2": print(" WARNING: The SCS-MP2 density is not implemented! Using unscaled MP2 density for property calculations.")
    

    if additional_print:
            
        print("Molecular orbital eigenvalues:\n")
        print("~~~~~~~~~~~~~~~~~~~~~~~")
        print(" N     Occ    Eps (Hart.)")
        print("~~~~~~~~~~~~~~~~~~~~~~~")
            
        if n_electrons > 1: occupancies = [2] * n_occ + [0] * int((len(epsilons) - n_occ))
        else: occupancies = [1] + [0] * (len(epsilons) - 1)
            
        for i in range(len(epsilons)):
            
            if i < 9: print(f"  {i + 1}     {occupancies[i]}     {np.round(epsilons[i],decimals=6)}")
            else: print(f" {i + 1}     {occupancies[i]}     {np.round(epsilons[i],decimals=6)}")
        print("~~~~~~~~~~~~~~~~~~~~~~")
        print("\n")

        symbol_list = []
        n_list = []
        switch_value = 0

        for state in range(len(epsilons)):
            
            if state == 0: print(f"Molecular orbital coefficients for ground state:\n")
            else: print(f"Molecular orbital coefficients for virtual state {state}:\n")
                
            for i, atom in enumerate(molecule):
                for j, orbital in enumerate(atom):
                    
                    symbol_list.append(atoms[i])                  
                    n_list.append(j + 1)
                    
                    if i == 1 and j == 0 and state == 0: switch_value = len(symbol_list) - 1
                
                
            for k in range(len(molecular_orbitals.T[state])):
                if k == switch_value: print("")
                print(" " + symbol_list[k].lower().capitalize() + f"  {n_list[k]}s  :  " + str(np.round(molecular_orbitals.T[state][k], decimals=4)))

            print("")
            

    
                
                
    ionisation_energy, electron_affinity, homo_lumo_gap = calculate_koopman_parameters(epsilons, n_occ)
        
        
            
    if type(electron_affinity) == np.float64: electron_affinity = np.round(electron_affinity,decimals=6)
    if type(homo_lumo_gap) == np.float64: homo_lumo_gap = np.round(homo_lumo_gap,decimals=6)
        
    print(f"\n Koopman's theorem ionisation energy: {ionisation_energy:.6f}")
    print(f" Koopman's theorem electron affinity: {electron_affinity}")
    print(f" Energy gap between HOMO and LUMO: {homo_lumo_gap}")

    if len(atoms) != 1:

        B_per_cm, B_GHz = calculate_rotational_constant(m_list, coordinates)
                
        print(f"\n Rotational constant (GHz): {B_GHz:.5f}")

        centre_of_mass = calculate_centre_of_mass(m_list, coordinates)

        print(f"\n Dipole moment origin is the centre of mass, {centre_of_mass:.4f} Angstroms.")
        D = integ.evaluate_dipole_integrals(orbitals, centre_of_mass)

        D_nuclear = calculate_nuclear_dipole_moment(centre_of_mass, Z_list, coordinates)        
        D_electronic = calculate_electronic_dipole_moment(P, D)

        total_dipole = D_nuclear + D_electronic

        print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("                Dipole Moment")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"  Nuclear: {D_nuclear:.5f}    Electronic: {D_electronic:.5f}\n")

        print(f"  Total: {total_dipole:.5f}",end="")
            
        if total_dipole > 0 or total_dipole < 0: 
            print("        " + molecular_structure, end="")
            print("  +--->")

        else: print(f"           {molecular_structure}")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        
    
        mull_qs, mull_qsum, mull_bo, low_qs, low_qsum, low_bo = population_analysis(P, S, coordinates, Z_list, atoms, ao_ranges)
        
            
        mull_f_qs = []; low_f_qs = []
            
        for i, q in enumerate(mull_qs):
            f_q = f"{q:.5f}"
            if q >= 0: f_q = " " + f_q 
            if atoms[i] == "H": f_q = " " + f_q
            mull_f_qs.append(f_q)
            
        for i, q in enumerate(low_qs):
            f_q = f"{q:.5f}"
            if q >= 0: f_q = " " + f_q 
            if atoms[i] == "H": f_q = " " + f_q
            low_f_qs.append(f_q)

        if mull_qsum < 0: space = " "
        else: space = "  "
        if mull_bo < 0: space2 = ""
        else: space2 = " "

        print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~        ~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("      Mulliken Charges                   Lowdin Charges")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~        ~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"  {atoms[0].lower().capitalize()}: {mull_f_qs[0]}                      {atoms[0].lower().capitalize()}: {low_f_qs[0]}        ")
        print(f"  {atoms[1].lower().capitalize()}: {mull_f_qs[1]}                      {atoms[1].lower().capitalize()}: {low_f_qs[1]}")
        print(f"\n  Sum of charges: {mull_qsum:.5f}   {space}      Sum of charges: {low_qsum:.5f}") 
        print(f"  Bond order: {mull_bo:.5f}      {space2}        Bond order: {low_bo:.5f}") 
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~        ~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return