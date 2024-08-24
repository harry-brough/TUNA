import numpy as np
import tuna_util as util

def format_output_line(E, delta_E, maxDP, rmsDP, damping_factor, step, orbital_gradient):

        delta_E_f = f"{delta_E:.10f}"
        if E >= 0: energy_f = " " + f"{E:.10f}"
        else: energy_f = f"{E:.10f}"
        
        if abs(delta_E) >= 10: delta_E_f = ""+ f"{delta_E:.10f}"
        if delta_E >= 0: delta_E_f = "  "+ f"{delta_E:.10f}"
        elif abs(delta_E) >= 0: delta_E_f = " "+ f"{delta_E:.10f}"    
        else: delta_E_f = f"{delta_E:.10f}"
        
        if abs(maxDP) >= 1000: maxDP_f = f"{maxDP:.10f}"
        elif abs(maxDP) >= 100: maxDP_f = " " + f"{maxDP:.10f}"
        elif abs(maxDP) >= 10: maxDP_f = "  " + f"{maxDP:.10f}"
        else: maxDP_f = "   "+f"{maxDP:.10f}"
        
        if abs(rmsDP) >= 1000: rmsDP_f = f"{rmsDP:.10f}"
        elif abs(rmsDP) >= 100: rmsDP_f = " "+f"{rmsDP:.10f}"
        elif abs(rmsDP) >= 10: rmsDP_f = "  "+ f"{rmsDP:.10f}"
        else: rmsDP_f = "   " +f"{rmsDP:.10f}"
        
        
        damping_factor_f = f"{damping_factor:.3f}"
        if damping_factor == 0: damping_factor_f = " ---"
        
        if step < 10: step_f = str(step) + " "
        else: step_f = str(step)
        if step != 1: print("")
        print(f"   {step_f}     {energy_f}     {delta_E_f}  {rmsDP_f}  {maxDP_f}     {orbital_gradient:.10f}     {damping_factor_f}",end="")   



def construct_density_matrix(mol_orbitals, n_occ, n_electrons_per_orbital):

    """Requires molecular orbitals and number of occupied orbitals, builds the density from these by tensor contraction.
    Factor of 2 two electrons per orbital, which has a factor of 0.5 with the density.
    Returns the one particle reduced density matrix."""

    P = n_electrons_per_orbital * np.einsum('io,jo->ij', mol_orbitals[:, :n_occ], mol_orbitals[:, :n_occ], optimize=True)

    return P
    


def diagonalise_Fock_matrix(F, X):

    F_orthonormal = X.T @ F @ X
    epsilons, eigenvectors = np.linalg.eigh(F_orthonormal)
    molecular_orbitals = X @ eigenvectors

    return epsilons, molecular_orbitals



def calculate_electronic_energy(P, H_Core, F):

    """Requires density, core Hamiltonian and Fock matrix. Contracts the density matrix with the total energy matrix into the
    electronic energy. Factor of 0.5 prevents overcounting electron-electron repulsion. Returns electronic energy."""

    electronic_energy = np.einsum('ij,ij->', 0.5 * P, H_Core + F, optimize=True)
    
    return electronic_energy



def calculate_UHF_electronic_energy(P_alpha, P_beta, H_Core, F_alpha, F_beta):

    electronic_energy = 0.5 * (np.einsum('ij,ij->', (P_alpha + P_beta), H_Core, optimize=True) + np.einsum('ij,ij->', P_alpha, F_alpha, optimize=True) + np.einsum('ij,ij->', P_beta, F_beta, optimize=True))
    
    return electronic_energy




def calculate_energy_components(P, T, V_NE, J, K):
    
    """Requires density, kinetic energy, nuclear attraction, Coulomb and exchange
    matrices. Uses tensor contraction with density matrix to calculate expectation
    values for each matrix, and returns these. Two-electron expectation values
    are halved to prevent overcounting."""

    kinetic_energy = np.einsum('ij,ij->', P, T, optimize=True)
    nuclear_electron_energy = np.einsum('ij,ij->', P, V_NE, optimize=True)
    coulomb_energy = 0.5 * np.einsum('ij,ij->', P, J, optimize=True)
    exchange_energy = -0.5 * np.einsum('ij,ij->', P, K / 2, optimize=True)

    return kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy
    


def calculate_SCF_changes(E, E_old, P, P_old):

    """Requires energy, density and energy and density from previous step.
    Calculates change in energy, and maximum and root-mean-square changes in
    density matrix, returns these values."""

    delta_E = E - E_old
    delta_P = P - P_old
    
    #Maximum and root-mean-square change in density matrix
    maxDP = np.max(delta_P)
    rmsDP = np.sqrt(np.mean(delta_P ** 2))

    return delta_E, maxDP, rmsDP



def construct_Fock_matrix(H_Core, V_EE, P):

    """Takes in the core Hamiltonian, two-electron integrals and density matrix. Uses tensor contraction
    to extract the coulomb (J) and exchange (K) integrals weighted by the density matrix, forming
    the two-electron contribution to the Fock matrix. Returns the Fock matrix, J and K."""

    #Forms two-electron contributions by tensor contraction of two-electron integrals with density matrix
    J = np.einsum('ijkl,kl->ij', V_EE, P, optimize=True)
    K = np.einsum('ilkj,kl->ij', V_EE, P, optimize=True)

    #Two-electron part of Fock matrix   
    G = J - 0.5 * K
    
    F = H_Core + G
    
    return F, J, K
    

def construct_UHF_Fock_matrix(H_Core, V_EE, P_alpha, P_beta):

    J_alpha = np.einsum('ijkl,kl->ij', V_EE, P_alpha, optimize=True)
    J_beta = np.einsum('ijkl,kl->ij', V_EE, P_beta, optimize=True)

    K_alpha = np.einsum('ilkj,kl->ij', V_EE, P_alpha, optimize=True)
    K_beta = np.einsum('ilkj,kl->ij', V_EE, P_beta, optimize=True)

    F_alpha = H_Core + (J_alpha + J_beta) - K_alpha
    F_beta = H_Core + (J_alpha + J_beta) - K_beta


    
    return F_alpha, F_beta, J_alpha, J_beta, K_alpha, K_beta
    
    

def damping(P, P_old, orbitalGrad, calculation):

    """Requires the current and old density matrix, the orbital gradient and calculation class. Checks if damping is enabled, then, uses
    custom formula to determine damping factor from DIIS error (orbital gradient) which only applies if the orbital gradient is sufficiently high, 
    otherwise the Fock matrix extrapolation (if DIIS is enabled) is allowed to accelerate convergence. If the slow or very slow convergence parameters
    are active, high static damping values are assigned. The density matrix is then mixed with the old one, with proportion determined by the damping factor,
    before the new density matrix and the damping factor are returned."""
    
    damping_factor = 0
    
    if calculation.damping:

        #Uses custom damping formula iff orbital gradient is high, otherwise allows DIIS to run 
        if orbitalGrad > 0.01: damping_factor = 0.7 * np.tanh(orbitalGrad)  

        if calculation.slowconv: damping_factor = 0.5
        elif calculation.veryslowconv: damping_factor = 0.85       

    #Mixes old density with new, in proportion of damping factor
    P_damped = damping_factor * P_old + (1 - damping_factor) * P
    
    return P_damped, damping_factor
        


def apply_level_shift(F, P, level_shift_parameter):

    """Requries Fock matrix, density matrix and level shift parameter. Updates Fock matrix
    to increase values of virtual orbital eigenvalues to increase convergence. Returns
    updated Fock matrix."""

    F_levelshift = F - level_shift_parameter * P

    return F_levelshift
    

def calculate_diis_error(F, P, S, X, Fock_vector, diis_error_vector):

    diis_error = np.einsum('ij,jk,kl->il', F, P, S, optimize=True) - np.einsum('ij,jk,kl->il', S, P, F, optimize=True)
    orthogonalised_diis_error = X.T @ diis_error @ X

    orbital_gradient = np.sqrt(np.mean(orthogonalised_diis_error ** 2))

    Fock_vector.append(F)
    diis_error_vector.append(orthogonalised_diis_error)

    return orbital_gradient, Fock_vector, diis_error_vector


def update_diis(Fock_vector, error_vector, F, X, n_doubly_occ, n_electrons_per_orbital, silent=False):

    dimension = len(Fock_vector) + 1
    B = np.empty((dimension, dimension))

    B[-1, :] = -1
    B[:, -1] = -1
    B[-1, -1] = 0

    for i in range(len(Fock_vector)):
        for j in range(len(Fock_vector)):

            B[i,j] = np.einsum("ij,ij->", error_vector[i], error_vector[j], optimize=True)


    right_hand_side = np.zeros((dimension))
    right_hand_side[-1] = -1

    try: 
        
        coeff = np.linalg.solve(B, right_hand_side)

        F_diis = np.zeros_like(F)

        for k in range(coeff.shape[0] - 1): F_diis += coeff[k] * Fock_vector[k]

        F_orthonormal_diis = X.T @ F_diis @ X
        molecular_orbitals_diis = X @ np.linalg.eigh(F_orthonormal_diis)[1]

        P = construct_density_matrix(molecular_orbitals_diis, n_doubly_occ, n_electrons_per_orbital)

        return P

    except np.linalg.LinAlgError: 
        
        if not silent: print("   (Resetting DIIS)", end="")

    return 0



def check_convergence(scf_conv, step, delta_E, maxDP, rmsDP, orbital_gradient, silent=False):

    if delta_E < scf_conv.get("delta_E") and maxDP < scf_conv.get("maxDP") and rmsDP < scf_conv.get("rmsDP") and orbital_gradient < scf_conv.get("orbitalGrad"): 
            
        if not silent:

            print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"\n Self-consistent field converged in {step} cycles!\n")

        return True    

    return False   




def SCF(molecule, calculation, T, V_NE, V_EE, V_NN, S, X, E_guess, P=None, P_alpha=None, P_beta=None, silent=False):


    if not silent:

        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("                                            SCF Cycle Iterations")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("  Step          E                  DE             RMS-DP           MAX-DP           [F,P]        Damping")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    H_Core = T + V_NE
    F = H_Core

    electronic_energy = E_guess

    level_shift_parameter = calculation.level_shift_parameter
    reference = calculation.reference
    maximum_iterations = calculation.max_iter
    level_shift = calculation.level_shift
    diis = calculation.diis
    scf_conv = calculation.scf_conv

    n_doubly_occ = molecule.n_doubly_occ
    n_alpha = molecule.n_alpha
    n_beta = molecule.n_beta
    n_electrons_per_orbital = calculation.n_electrons_per_orbital

    orbital_gradient = 1
    level_shift_off = False


    if reference == "RHF":

        Fock_vector = []
        diis_error_vector = []

        for step in range(1, maximum_iterations):

            electronic_energy_old = electronic_energy
            P_old = P 
      
            F, J, K = construct_Fock_matrix(H_Core, V_EE, P)

            orbital_gradient, Fock_vector, diis_error_vector = calculate_diis_error(F, P, S, X, Fock_vector, diis_error_vector)

            epsilons, molecular_orbitals = diagonalise_Fock_matrix(F, X)
            
            P = construct_density_matrix(molecular_orbitals, n_doubly_occ, n_electrons_per_orbital)
            electronic_energy = calculate_electronic_energy(P, H_Core, F)

            delta_E, maxDP, rmsDP = calculate_SCF_changes(electronic_energy, electronic_energy_old, P, P_old)


            if level_shift and not level_shift_off:

                if orbital_gradient > 0.00001: F = apply_level_shift(F, P, level_shift_parameter)

                else: 

                    level_shift_off = True
                    if not silent: print("    (Level Shift Off)", end="")


            if len(Fock_vector) > 10: del Fock_vector[0]; del diis_error_vector[0]
  
            if step > 2 and diis and orbital_gradient < 0.2 and orbital_gradient > 1e-5: 
                
                P_diis = update_diis(Fock_vector, diis_error_vector, F, X, n_doubly_occ, n_electrons_per_orbital, silent=silent)
                if type(P_diis) != int: P = P_diis

            P, damping_factor = damping(P, P_old, orbital_gradient, calculation)

            E = electronic_energy + V_NN  

            if not silent: format_output_line(E, delta_E, maxDP, rmsDP, damping_factor, step, orbital_gradient)

            if check_convergence(scf_conv, step, delta_E, maxDP, rmsDP, orbital_gradient, silent=silent): 

                kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy = calculate_energy_components(P, T, V_NE, J, K)
                  

                return E, molecular_orbitals, epsilons, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy, P, P_alpha, P_beta


    elif reference == "UHF":

        Fock_vector_alpha = []
        Fock_vector_beta = []

        diis_error_vector_alpha = []
        diis_error_vector_beta = []

        P = P_alpha + P_beta 


        for step in range(1, maximum_iterations):

            electronic_energy_old = electronic_energy
            P_old_alpha = P_alpha
            P_old_beta = P_beta
            P_old = P

            F_alpha, F_beta, J_alpha, J_beta, K_alpha, K_beta = construct_UHF_Fock_matrix(H_Core, V_EE, P_alpha, P_beta)

            orbital_gradient_alpha, Fock_vector_alpha, diis_error_vector_alpha = calculate_diis_error(F_alpha, P_alpha, S, X, Fock_vector_alpha, diis_error_vector_alpha)
            orbital_gradient_beta, Fock_vector_beta, diis_error_vector_beta = calculate_diis_error(F_beta, P_beta, S, X, Fock_vector_beta, diis_error_vector_beta)

            epsilons_alpha, molecular_orbitals_alpha = diagonalise_Fock_matrix(F_alpha, X)
            epsilons_beta, molecular_orbitals_beta = diagonalise_Fock_matrix(F_beta, X)

            P_alpha = construct_density_matrix(molecular_orbitals_alpha, n_alpha, n_electrons_per_orbital)
            P_beta = construct_density_matrix(molecular_orbitals_beta, n_beta, n_electrons_per_orbital)

            electronic_energy = calculate_UHF_electronic_energy(P_alpha, P_beta, H_Core, F_alpha, F_beta)

            P = P_alpha + P_beta

            delta_E, maxDP, rmsDP = calculate_SCF_changes(electronic_energy, electronic_energy_old, P, P_old)


            if level_shift and not level_shift_off:

                if min(orbital_gradient_alpha, orbital_gradient_beta) > 0.00001:

                    F_alpha = apply_level_shift(F_alpha, P_alpha, level_shift_parameter)
                    F_beta = apply_level_shift(F_beta, P_beta, level_shift_parameter)

                else: 

                    level_shift_off = True
                    if not silent: print("    (Level Shift Off)", end="")


            if len(Fock_vector_alpha) > 10: del Fock_vector_alpha[0]; del diis_error_vector_alpha[0]
            if len(Fock_vector_beta) > 10: del Fock_vector_beta[0]; del diis_error_vector_beta[0]

            if step > 2 and diis and orbital_gradient_alpha < 0.001 and orbital_gradient_alpha > 1e-5 and orbital_gradient_beta < 0.001 and orbital_gradient_beta > 1e-5: 

                P_diis_alpha = update_diis(Fock_vector_alpha, diis_error_vector_alpha, F_alpha, X, n_alpha, n_electrons_per_orbital, silent=silent)
                P_diis_beta = update_diis(Fock_vector_beta, diis_error_vector_beta, F_beta, X, n_beta, n_electrons_per_orbital, silent=silent)

                if type(P_diis_alpha) != int: 
                    P_alpha = P_diis_alpha
                else: Fock_vector_alpha = []
                if type(P_diis_beta) != int: P_beta = P_diis_beta

   
            orbital_gradient = orbital_gradient_alpha + orbital_gradient_beta

            P_alpha, damping_factor = damping(P_alpha, P_old_alpha, orbital_gradient_alpha, calculation)
            P_beta, damping_factor = damping(P_beta, P_old_beta, orbital_gradient_beta, calculation)

            E = electronic_energy + V_NN  

            if not silent: format_output_line(E, delta_E, maxDP, rmsDP, damping_factor, step, orbital_gradient)

            if check_convergence(scf_conv, step, delta_E, maxDP, rmsDP, orbital_gradient, silent=silent): 

                kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy = calculate_energy_components(P, T, V_NE, J_alpha + J_beta, 2 * (K_alpha + K_beta))
                
                epsilons_combined = np.concatenate((epsilons_alpha, epsilons_beta))
                molecular_orbitals_combined = np.concatenate((molecular_orbitals_alpha, molecular_orbitals_beta), axis=1)

                epsilons = epsilons_combined[np.argsort(epsilons_combined)]
                molecular_orbitals = molecular_orbitals_combined[:, np.argsort(epsilons_combined)]


                return E, molecular_orbitals, epsilons, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy, P, P_alpha, P_beta

   
    util.error(f"Self-consistent field not converged in {maximum_iterations} iterations! Increase maximum iterations or give up.")

