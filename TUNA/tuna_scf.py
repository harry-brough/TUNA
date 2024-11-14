import numpy as np
from tuna_util import *


def format_output_line(E, delta_E, maxDP, rmsDP, damping_factor, step, orbital_gradient, calculation):

    """
    
    Requires energy (float), change in energy (energy), maximum change in the density matrix (float), root-mean-square change in the density matrix (float),
    damping factor (float), step number (int), orbital gradient (float) and calculation (Calculation).

    Formats everything nicely and prints to the output line to make sure decimal points all stay aligned properly.

    """

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
    if step != 1: log("", calculation, 1)


    log(f"   {step_f}     {energy_f}     {delta_E_f}  {rmsDP_f}  {maxDP_f}     {orbital_gradient:.10f}     {damping_factor_f}", calculation, 1, end="")   



def construct_density_matrix(molecular_orbitals, n_occ, n_electrons_per_orbital):

    """
    
    Requires molecular orbitals (array), number of occupied orbitals (int), and number of electrons per orbital (int).

    The prefactor indicates whether the calculation is RHF and UHF, so either two or one electrons per orbital. Builds the 
    density from these and molecular orbitals by tensor contraction.

    Returns the one particle reduced density matrix (array).
    
    """

    P = n_electrons_per_orbital * np.einsum('io,jo->ij', molecular_orbitals[:, :n_occ], molecular_orbitals[:, :n_occ], optimize=True)

    return P
    


def diagonalise_Fock_matrix(F, X):

    """
    
    Requires Fock matrix (array) and Fock transformation matrix (array).

    Transforms Fock matrix into orthonormal basis to calculate molecular orbitals and orbital eigenvalues.

    Returns epsilons (array) and molecular orbitals (array).
    
    """

    F_orthonormal = X.T @ F @ X
    epsilons, eigenvectors = np.linalg.eigh(F_orthonormal)
    molecular_orbitals = X @ eigenvectors

    return epsilons, molecular_orbitals



def calculate_RHF_electronic_energy(P, H_Core, G):

    """
    
    Requires the density matrix (array), core Hamiltonian matrix (array) and two-electron part of Fock matrix (array).

    Calculates and returns the electronic energy (float) from optimised tensor contraction of density matrix and Fock matrix.

    """

    electronic_energy = np.einsum('ij,ij->', 0.5 * P, H_Core + G, optimize=True)
    
    return electronic_energy



def calculate_UHF_electronic_energy(P_alpha, P_beta, H_Core, F_alpha, F_beta):

    """
    
    Requires the alpha and beta density matrices (arrays), core Hamiltonian matrix (array) and alpha and beta Fock matrices (arrays).

    Calculates and returns the electronic energy (float) from optimised tensor contraction of density matrices and Fock matrices.

    """

    electronic_energy = 0.5 * (np.einsum('ij,ij->', (P_alpha + P_beta), H_Core, optimize=True) + np.einsum('ij,ij->', P_alpha, F_alpha, optimize=True) + np.einsum('ij,ij->', P_beta, F_beta, optimize=True))
    
    return electronic_energy




def calculate_UHF_energy_components(P_alpha, P_beta, T, V_NE, J_alpha, J_beta, K_alpha, K_beta):
    
    """
    
    Requires alpha and beta density matrices (arrays), kinetic matrix (array), nuclear-electron potential energy matrix (array),
    alpha and beta Coulomb and exchange matrices (arrays).
    
    Calculates various energy components through optimised tensor contractions.
    
    Returns kinetic energy (float), nuclear-electron energy (float), Coulomb energy (float) and exchange energy (float).
    
    """

    kinetic_energy = np.einsum('ij,ij->', P_alpha + P_beta, T, optimize=True)
    nuclear_electron_energy = np.einsum('ij,ij->', P_alpha + P_beta, V_NE, optimize=True)
    coulomb_energy = 0.5 * np.einsum('ij,ij->', P_alpha + P_beta, J_alpha + J_beta, optimize=True) 
    exchange_energy = -0.5 * np.einsum('ij,ij->', P_alpha, K_alpha, optimize=True) - 0.5 * np.einsum('ij,ij->', P_beta, K_beta, optimize=True)

    return kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy



def calculate_RHF_energy_components(P, T, V_NE, J, K):
    
    """
    
    Requires density matrix (array), kinetic matrix (array), nuclear-electron potential energy matrix (array), Coulomb and exchange matrices (arrays).
    
    Calculates various energy components through optimised tensor contractions.
    
    Returns kinetic energy (float), nuclear-electron energy (float), Coulomb energy (float) and exchange energy (float).
    
    """

    kinetic_energy = np.einsum('ij,ij->', P, T, optimize=True)
    nuclear_electron_energy = np.einsum('ij,ij->', P, V_NE, optimize=True)
    coulomb_energy = 0.5 * np.einsum('ij,ij->', P, J, optimize=True)
    exchange_energy = -0.5 * np.einsum('ij,ij->', P, K / 2, optimize=True)

    return kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy
    


def calculate_SCF_changes(E, E_old, P, P_old):

    """
    
    Requires energy (float), old energy (float), new and old density matrices (array).

    Calculates the maximum and root-mean-square change in the density matrix, from the old one, and the change in the energy.

    Returns change in energy (float), maximum change in the density matrix (float) and root-mean-square change in the density matrix (array).
    
    """

    delta_E = E - E_old
    delta_P = P - P_old
    
    #Maximum and root-mean-square change in density matrix
    maxDP = np.max(delta_P)
    rmsDP = np.sqrt(np.mean(delta_P ** 2))

    return delta_E, maxDP, rmsDP



def construct_RHF_Fock_matrix(H_Core, V_EE, P):

    """
    
    Requires core Hamiltonian matrix (array), two-electron integral matrix (array) and density matrix (array).

    Calculates the two-electron contributions (Coulomb and exchange matrices) by tensor contraction.

    Returns Fock matrix, Coulomb matrix and exchange matrix (arrays).
    
    """

    #Forms two-electron contributions by tensor contraction of two-electron integrals with density matrix
    J = np.einsum('ijkl,kl->ij', V_EE, P, optimize=True)
    K = np.einsum('ilkj,kl->ij', V_EE, P, optimize=True)

    #Two-electron part of Fock matrix   
    G = J - 0.5 * K
    
    F = H_Core + G
    
    return F, J, K
    



def construct_UHF_Fock_matrix(H_Core, V_EE, P_alpha, P_beta):

    """
    
    Requires core Hamiltonian matrix (array), two-electron integral matrix (array) and alpha and beta density matrices (array).

    Calculates the two-electron contributions (Coulomb and exchange matrices) for both alpha and beta electrons by tensor contraction.

    Returns Fock matrices, Coulomb matrices and exchange matrices (arrays).
    
    """

    J_alpha = np.einsum('ijkl,kl->ij', V_EE, P_alpha, optimize=True)
    J_beta = np.einsum('ijkl,kl->ij', V_EE, P_beta, optimize=True)

    K_alpha = np.einsum('ilkj,kl->ij', V_EE, P_alpha, optimize=True)
    K_beta = np.einsum('ilkj,kl->ij', V_EE, P_beta, optimize=True)

    #Builds separate Fock matrices for alpha and beta spins
    F_alpha = H_Core + (J_alpha + J_beta) - K_alpha
    F_beta = H_Core + (J_alpha + J_beta) - K_beta


    return F_alpha, F_beta, J_alpha, J_beta, K_alpha, K_beta
    
    


def damping(P, P_old, orbital_gradient, calculation):

    """
    
    Requires density matrix (array), old density matrix (array), orbital gradient (float), calculation (Calculation).

    If damping has been requested, calculates a damping factor based on a homemade equation, and if SLOWCONV has been requested,
    a high static damping factor is required. If VERYSLOWCONV is requested, a very high static damping factor is used. The new
    density matrix is then damped by mixing with the old density matrix in this proportion.

    Returns damped density matrix (array) and final damping factor (float).

    """
    
    damping_factor = 0
    
    if calculation.damping:

        #Uses custom damping formula iff orbital gradient is high, otherwise allows DIIS to run 
        if orbital_gradient > 0.01: damping_factor = 0.7 * np.tanh(orbital_gradient)  

        if calculation.slowconv: damping_factor = 0.5
        elif calculation.veryslowconv: damping_factor = 0.85       

    #Mixes old density with new, in proportion of damping factor
    P_damped = damping_factor * P_old + (1 - damping_factor) * P
    

    return P_damped, damping_factor
        



def apply_level_shift(F, P, level_shift_parameter):

    """
    
    Requires Fock matrix (array), density matrix (array) and level shift parameter (float).

    The level shift is applied by multiplying the density matrix and subtracting from the Fock matrix.

    Returns the level-shifted Fock matrix.
    
    """

    F_levelshift = F - level_shift_parameter * P

    return F_levelshift
    


def calculate_diis_error(F, P, S, X, Fock_vector, diis_error_vector):

    """
    
    Requires Fock matrix (array), density matrix (array), overlap matrix (array) and Fock transformation matrix (array), and DIIS error vector (array).

    Calculates the DIIS error, then orthogonalises this and calculates the root-mean-square error, which is called the orbital gradient. The Fock matrix
    is then appended onto the Fock vector and the DIIS error is appended to the DIIS error vector.

    Returns the orbital gradient (float), Fock vector (array) and DIIS error vector (array).

    """

    diis_error = np.einsum('ij,jk,kl->il', F, P, S, optimize=True) - np.einsum('ij,jk,kl->il', S, P, F, optimize=True)
    orthogonalised_diis_error = X.T @ diis_error @ X

    #Orbital gradient is the root-mean-squared DIIS error
    orbital_gradient = np.sqrt(np.mean(orthogonalised_diis_error ** 2))

    #Updates vectors with Fock and error matrices
    Fock_vector.append(F)
    diis_error_vector.append(orthogonalised_diis_error)


    return orbital_gradient, Fock_vector, diis_error_vector



def update_diis(Fock_vector, error_vector, F, X, n_doubly_occ, n_electrons_per_orbital, calculation, silent=False):

    """
    
    Requires Fock vector and error vector (arrays), current Fock matrix (array), Fock transformation matrix (array), number of doubly occupied orbitals
    (int), number of electrons per orbitals (int) and calculation (Calculation).

    Builds the system of equations needed to solve the DIIS coefficients, then uses these to build a DIIS extrapolated Fock matrix, and diagonalises this
    for molecular orbitals to build a new density matrix.

    If DIIS equations can be solved, returns the new density matrix (array), otherwise, returns 0 and outputs a warning, resetting DIIS and clearing the error vector.
    
    """

    dimension = len(Fock_vector) + 1
    B = np.empty((dimension, dimension))

    B[-1, :] = -1
    B[:, -1] = -1
    B[-1, -1] = 0

    #Builds B matrix involved in linear DIIS equations
    for i in range(len(Fock_vector)):
        for j in range(len(Fock_vector)):

            B[i,j] = np.einsum("ij,ij->", error_vector[i], error_vector[j], optimize=True)


    right_hand_side = np.zeros((dimension))
    right_hand_side[-1] = -1


    try: 
        
        #Solves system of linear equations for DIIS coefficients
        coeff = np.linalg.solve(B, right_hand_side)

        F_diis = np.zeros_like(F)

        #Builds extrapolated Fock matrix using the calculated DIIS coefficients
        for k in range(coeff.shape[0] - 1): F_diis += coeff[k] * Fock_vector[k]

        #Diagonalises extrapolated Fock matrix for molecular orbitals, then uses these to build the new density matrix
        F_orthonormal_diis = X.T @ F_diis @ X
        molecular_orbitals_diis = X @ np.linalg.eigh(F_orthonormal_diis)[1]

        P = construct_density_matrix(molecular_orbitals_diis, n_doubly_occ, n_electrons_per_orbital)

        return P

    except np.linalg.LinAlgError: 

        #If system of equations can't be solved, reset DIIS with this warning
        if not silent: log("   (Resetting DIIS)", calculation, 1, end="")

    return 0



def check_convergence(scf_conv, step, delta_E, maxDP, rmsDP, orbital_gradient, calculation, silent=False):

    """
    
    Requires SCF convergence criteria (dict), step (int), change in energy (float), maximum change in the density matrix (float), root-mean-square change in the density
    matrix (float), orbital gradient (float), and calculation (Calculation).
    
    Checks convergence of various quantities against criteria, returns True if converged and prints out positive message, or False if not yet converged.

    """
    
    #Checks various quantities against convergence criteria
    if np.abs(delta_E) < scf_conv.get("delta_E") and np.abs(maxDP) < scf_conv.get("maxDP") and np.abs(rmsDP) < scf_conv.get("rmsDP") and np.abs(orbital_gradient) < scf_conv.get("orbitalGrad"): 

        if not silent:

            log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
            log(f"\n Self-consistent field converged in {step} cycles!\n", calculation, 1)

        return True    

    return False   




def SCF(molecule, calculation, T, V_NE, V_EE, V_NN, S, X, E_guess, P=None, P_alpha=None, P_beta=None, silent=False):

    """
    
    Requires molecule (Molecule), calculation (Calculation), kinetic energy matrix (array), nuclear-electron attraction matrix (array), electron-electron
    repulsion matrix (array), nuclear-nuclear repulsion energy (float), overlap matrix (array), Fock transformation matrix (array) and energy guess (float).
    Optional parameters are guess density matrices (array) and whether any information should print to the console.    
    
    Runs the SCF loop, which depends on whether the reference is RHF or UHF. Constructs the Fock matrix, diagonalises the Fock matrix, builds the density
    matrix then checks for convergence. This process then interates until convergence is achieved. Various convergence acceleration methods are optionally
    included, such as level shift, damping and DIIS. In the end, packages all the useful information from the converged SCF cycle into an Output object.

    Returns the SCF output (Output), kinetic energy (float), nuclear-electron energy (float), Coulomb energy (float) and exchange energy (float).

    """

    if not silent:

        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        log("                                            SCF Cycle Iterations", calculation, 1)
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        log("  Step          E                  DE             RMS(DP)          MAX(DP)          [F,PS]       Damping", calculation, 1)
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)


    #Build core Hamiltonian and initial Fock matrix
    H_Core = T + V_NE
    F = H_Core

    #Sets initial parameters
    electronic_energy = E_guess
    orbital_gradient = 1
    level_shift_off = False

    #Unpacks useful calculation properties
    level_shift_parameter = calculation.level_shift_parameter
    reference = calculation.reference
    maximum_iterations = calculation.max_iter
    level_shift = calculation.level_shift
    diis = calculation.diis
    scf_conv = calculation.scf_conv
    n_electrons_per_orbital = calculation.n_electrons_per_orbital

    #Unpacks useful molecular quantities
    n_doubly_occ = molecule.n_doubly_occ
    n_alpha = molecule.n_alpha
    n_beta = molecule.n_beta


    if reference == "RHF":

        #Initialises vectors for DIIS
        Fock_vector = []
        diis_error_vector = []

        for step in range(1, maximum_iterations):
            
            #Sets previous values to current value to initialise loop for comparison later
            electronic_energy_old = electronic_energy
            P_old = P 
      
            #Constructs Fock matrix
            F, J, K = construct_RHF_Fock_matrix(H_Core, V_EE, P)

            #Calculates DIIS error
            orbital_gradient, Fock_vector, diis_error_vector = calculate_diis_error(F, P, S, X, Fock_vector, diis_error_vector)

            #Diagonalises Fock matrix
            epsilons, molecular_orbitals = diagonalise_Fock_matrix(F, X)
            
            #Constructs density matrix
            P = construct_density_matrix(molecular_orbitals, n_doubly_occ, n_electrons_per_orbital)

            #Calculates electronic energy
            electronic_energy = calculate_RHF_electronic_energy(P, H_Core, F)

            #Calculates the changes in energy and density
            delta_E, maxDP, rmsDP = calculate_SCF_changes(electronic_energy, electronic_energy_old, P, P_old)

            #Applies level shift
            if level_shift and not level_shift_off:

                if orbital_gradient > 0.00001: F = apply_level_shift(F, P, level_shift_parameter)

                else: 

                    level_shift_off = True
                    if not silent: log("    (Level Shift Off)", calculation, 1, end="")

            #Clears old Fock matrices if Fock vector is too old
            if len(Fock_vector) > 10: 
                
                del Fock_vector[0]
                del diis_error_vector[0]
  
            #Updates density matrix from DIIS extrapolated Fock matrix, applies it if the equations were solved successfully
            if step > 2 and diis and orbital_gradient < 0.2 and orbital_gradient > 1e-5: 
                
                P_diis = update_diis(Fock_vector, diis_error_vector, F, X, n_doubly_occ, n_electrons_per_orbital, calculation, silent=silent)
                if type(P_diis) != int: P = P_diis

            #Damping factor is applied to the density matrix
            P, damping_factor = damping(P, P_old, orbital_gradient, calculation)

            #Energy is sum of electronic and nuclear energies
            E = electronic_energy + V_NN  

            #Data outputted to console
            if not silent: format_output_line(E, delta_E, maxDP, rmsDP, damping_factor, step, orbital_gradient, calculation)

            #Check for convergence of energy and density
            if check_convergence(scf_conv, step, delta_E, maxDP, rmsDP, orbital_gradient, calculation, silent=silent): 

                kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy = calculate_RHF_energy_components(P, T, V_NE, J, K)
                  
                molecular_orbitals_alpha = molecular_orbitals
                molecular_orbitals_beta = molecular_orbitals

                epsilons_alpha = epsilons
                epsilons_beta = epsilons

                P_alpha = P / 2
                P_beta = P / 2
                
                #Builds SCF output object
                scf_output = Output(E, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta, epsilons, epsilons_alpha, epsilons_beta)
               
                return scf_output, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy 


    elif reference == "UHF":

        #Initialises vectors for DIIS
        Fock_vector_alpha = []
        Fock_vector_beta = []

        diis_error_vector_alpha = []
        diis_error_vector_beta = []

        P = P_alpha + P_beta 


        for step in range(1, maximum_iterations):

            #Sets previous values to current value to initialise loop for comparison later
            electronic_energy_old = electronic_energy
            P_old_alpha = P_alpha
            P_old_beta = P_beta
            P_old = P

            #Constructs Fock matrices
            F_alpha, F_beta, J_alpha, J_beta, K_alpha, K_beta = construct_UHF_Fock_matrix(H_Core, V_EE, P_alpha, P_beta)

            #Calculates DIIS error
            orbital_gradient_alpha, Fock_vector_alpha, diis_error_vector_alpha = calculate_diis_error(F_alpha, P_alpha, S, X, Fock_vector_alpha, diis_error_vector_alpha)
            orbital_gradient_beta, Fock_vector_beta, diis_error_vector_beta = calculate_diis_error(F_beta, P_beta, S, X, Fock_vector_beta, diis_error_vector_beta)

            #Diagonalises Fock matrices 
            epsilons_alpha, molecular_orbitals_alpha = diagonalise_Fock_matrix(F_alpha, X)
            epsilons_beta, molecular_orbitals_beta = diagonalise_Fock_matrix(F_beta, X)

            #Constructs density matrices
            P_alpha = construct_density_matrix(molecular_orbitals_alpha, n_alpha, n_electrons_per_orbital)
            P_beta = construct_density_matrix(molecular_orbitals_beta, n_beta, n_electrons_per_orbital)

            #Calculates electronic energy
            electronic_energy = calculate_UHF_electronic_energy(P_alpha, P_beta, H_Core, F_alpha, F_beta)

            P = P_alpha + P_beta

            #Calculates the changes in energy and density
            delta_E, maxDP, rmsDP = calculate_SCF_changes(electronic_energy, electronic_energy_old, P, P_old)

            #Applies level shift
            if level_shift and not level_shift_off:

                if min(orbital_gradient_alpha, orbital_gradient_beta) > 0.00001:

                    F_alpha = apply_level_shift(F_alpha, P_alpha, level_shift_parameter)
                    F_beta = apply_level_shift(F_beta, P_beta, level_shift_parameter)

                else: 

                    level_shift_off = True
                    if not silent: log("    (Level Shift Off)", calculation, 1, end="")

            #Clears old Fock matrices if Fock vector is too old
            if len(Fock_vector_alpha) > 10: 
                
                del Fock_vector_alpha[0]
                del diis_error_vector_alpha[0]

            if len(Fock_vector_beta) > 10:

                del Fock_vector_beta[0]
                del diis_error_vector_beta[0]

            #Updates density matrix from DIIS extrapolated Fock matrix, applies it if the equations were solved successfully
            if step > 2 and diis and orbital_gradient_alpha < 0.02 and orbital_gradient_alpha > 1e-5 and orbital_gradient_beta < 0.02 and orbital_gradient_beta > 1e-5: 

                P_diis_alpha = update_diis(Fock_vector_alpha, diis_error_vector_alpha, F_alpha, X, n_alpha, n_electrons_per_orbital, calculation, silent=silent)
                P_diis_beta = update_diis(Fock_vector_beta, diis_error_vector_beta, F_beta, X, n_beta, n_electrons_per_orbital, calculation, silent=silent)

                if type(P_diis_alpha) != int: P_alpha = P_diis_alpha
                else: Fock_vector_alpha = []
                
                if type(P_diis_beta) != int: P_beta = P_diis_beta
                else: Fock_vector_beta = []
   
   
            orbital_gradient = orbital_gradient_alpha + orbital_gradient_beta

            #Damping factor is applied to the density matrix
            P_alpha, damping_factor = damping(P_alpha, P_old_alpha, orbital_gradient_alpha, calculation)
            P_beta, damping_factor = damping(P_beta, P_old_beta, orbital_gradient_beta, calculation)

            #Energy is sum of electronic and nuclear energies
            E = electronic_energy + V_NN  

            #Outputs useful information to console
            if not silent: format_output_line(E, delta_E, maxDP, rmsDP, damping_factor, step, orbital_gradient, calculation)

            #Check for convergence of energy and density
            if check_convergence(scf_conv, step, delta_E, maxDP, rmsDP, orbital_gradient, calculation, silent=silent): 

                kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy = calculate_UHF_energy_components(P_alpha, P_beta, T, V_NE, J_alpha, J_beta, K_alpha,K_beta)
                
                epsilons_combined = np.concatenate((epsilons_alpha, epsilons_beta))
                molecular_orbitals_combined = np.concatenate((molecular_orbitals_alpha, molecular_orbitals_beta), axis=1)

                epsilons = epsilons_combined[np.argsort(epsilons_combined)]
                molecular_orbitals = molecular_orbitals_combined[:, np.argsort(epsilons_combined)]

                #Builds SCF Output object with useful quantities
                scf_output = Output(E, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta, epsilons, epsilons_alpha, epsilons_beta)

                return scf_output, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy
            

    error(f"Self-consistent field not converged in {maximum_iterations} iterations! Increase maximum iterations or give up.")

