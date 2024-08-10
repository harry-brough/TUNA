import numpy as np
import sys

def format_output_line(E, delta_E, maxDP, rmsDP, damping_factor, step, orbital_gradient):

        delta_E_f = f"{delta_E:.9f}"
        if E >= 0: energy_f = " " + f"{E:.9f}"
        else: energy_f = f"{E:.9f}"
        
        if abs(delta_E) >= 10: delta_E_f = ""+ f"{delta_E:.9f}"
        if delta_E >= 0: delta_E_f = "  "+ f"{delta_E:.9f}"
        elif abs(delta_E) >= 0: delta_E_f = " "+ f"{delta_E:.9f}"    
        else: delta_E_f = f"{delta_E:.9f}"
        
        if abs(maxDP) >= 1000: maxDP_f = f"{maxDP:.9f}"
        elif abs(maxDP) >= 100: maxDP_f = " " + f"{maxDP:.9f}"
        elif abs(maxDP) >= 10: maxDP_f = "  " + f"{maxDP:.9f}"
        else: maxDP_f = "   "+f"{maxDP:.9f}"
        
        if abs(rmsDP) >= 1000: rmsDP_f = f"{rmsDP:.9f}"
        elif abs(rmsDP) >= 100: rmsDP_f = " "+f"{rmsDP:.9f}"
        elif abs(rmsDP) >= 10: rmsDP_f = "  "+ f"{rmsDP:.9f}"
        else: rmsDP_f = "   " +f"{rmsDP:.9f}"
        
        
        damping_factor_f = f"{damping_factor:.3f}"
        if damping_factor == 0: damping_factor_f = " ---"
        
        step +=1
        if step < 10: step_f = str(step) + " "
        else: step_f = str(step)
        if step != 1: print("")
        print(f"   {step_f}     {energy_f}     {delta_E_f}  {rmsDP_f}  {maxDP_f}     {orbital_gradient:.7f}     {damping_factor_f}",end="")   



def construct_density_matrix(mol_orbitals, n_occ):

    P = 2 * np.einsum('io,jo->ij', mol_orbitals[:, :n_occ], mol_orbitals[:, :n_occ], optimize=True)

    return P
    
    
def calculate_electronic_energy(P, H_Core, F):

    electronic_energy = np.einsum('ij,ij->', 0.5 * P, H_Core + F, optimize=True)
    
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
    




def calculate_SCF_convergence(E, E_old, P, P_old):

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
    
    

def damping(P, P_old, step, orbitalGrad, DIIS_on, damping_on, slowconv, veryslowconv):
    
    damping_factor = 0
    
    if damping_on:
        
        if step < 2 or not DIIS_on: damping_factor = 0.7 * np.tanh(orbitalGrad)
        if slowconv: damping_factor = 0.7
        elif veryslowconv: damping_factor = 0.95

    P = damping_factor * P_old + (1 - damping_factor) * P
    
    return P, damping_factor
        


def level_shift(F, P, level_shift_parameter):

    """Requries Fock matrix, density matrix and level shift parameter. Updates Fock matrix
    to increase values of virtual orbital eigenvalues to increase convergence. Returns
    updated Fock matrix."""

    F_levelshift = F - level_shift_parameter * P

    return F_levelshift
    

def diis(Fock_vector, error_vector, F, X, n_occ):

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

    try: coeff = np.linalg.solve(B, right_hand_side)
    except np.linalg.LinAlgError: sys.exit("ERROR: The DIIS equations could not be solved!  :(")

    F_diis = np.zeros_like(F)

    for k in range(coeff.shape[0] - 1): F_diis += coeff[k] * Fock_vector[k]

    F_orthonormal_diis = X.T @ F_diis @ X
    epsilons_diis, eigenvectors_diis = np.linalg.eigh(F_orthonormal_diis)
    molecular_orbitals_diis = X @ eigenvectors_diis

    P = construct_density_matrix(molecular_orbitals_diis, n_occ)

    return P


def SCF(X, T, V_NE, V_EE, P, energy_guess, S, max_iter, n_electrons, V_NN, scf_conv, DIIS_on,level_shift_on, damping_on, slowconv, veryslowconv,silent=False):

    if not silent:
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("                                      SCF Cycle Iterations")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("  Step          E                 DE            RMS-DP          MAX-DP         [F,P]      Damping")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    H_Core = T + V_NE
    F = H_Core
    
    electronic_energy = energy_guess
    level_shift_parameter = 0.2
    orbital_gradient = 1
    level_shift_off = False
    n_occ = int(n_electrons / 2)

    Fock_vector = []
    error_vector = []
    
    for step in range(max_iter):
        
        electronic_energy_old = electronic_energy
        P_old = P

        F, J, K = construct_Fock_matrix(H_Core, V_EE, P)
        

        error = np.einsum('ij,jk,kl->il', F, P, S, optimize=True) - np.einsum('ij,jk,kl->il', S, P, F, optimize=True)
        orthogonalised_error = X.T @ error @ X
        orbital_gradient = np.sqrt(np.mean(orthogonalised_error ** 2))

        
        Fock_vector.append(F)
        error_vector.append(orthogonalised_error)



        
        
        F_orthonormal = X.T @ F @ X
        epsilons, eigenvectors = np.linalg.eigh(F_orthonormal)
        molecular_orbitals = X @ eigenvectors
        
        P = construct_density_matrix(molecular_orbitals, n_occ)
        electronic_energy = calculate_electronic_energy(P, H_Core, F)
        #try removing this P instead for level shift
        delta_E, maxDP, rmsDP = calculate_SCF_convergence(electronic_energy, electronic_energy_old, P, P_old)
        if level_shift_on and not level_shift_off:
            if orbital_gradient > 0.00001:
                F = level_shift(F, P, level_shift_parameter)
            else: 
                level_shift_off = True
                if not silent:
                    print("    (Level Shift Off)", end="")
        E = electronic_energy + V_NN
        
        if len(Fock_vector) > 10:
            del Fock_vector[0]
            del error_vector[0]


        if step > 1 and DIIS_on and orbital_gradient < 0.2 and orbital_gradient > 1e-5: 
            
            P = diis(Fock_vector, error_vector, F, X, n_occ)


        P, damping_factor = damping(P, P_old, step, orbital_gradient, DIIS_on, damping_on, slowconv, veryslowconv)


        if not silent: format_output_line(E, delta_E, maxDP, rmsDP, damping_factor, step, orbital_gradient)
        

        if delta_E < scf_conv.get("delta_E") and maxDP < scf_conv.get("maxDP") and rmsDP < scf_conv.get("rmsDP") and orbital_gradient < scf_conv.get("orbitalGrad"): 
            
            if not silent:
                print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("\n Self-consistent field converged!\n")
            
            kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy = calculate_energy_components(P, T, V_NE, J, K)
            
            return E, molecular_orbitals, epsilons, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy, P
        
    if not silent:
        print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        sys.exit(f"ERROR: Self-consistent field not converged in {max_iter} iterations! Increase maximum iterations or give up.  :(")
    sys.exit(f"ERROR: Gradient calculation failed to converge!  :(")

    return E, molecular_orbitals, epsilons, J, K
