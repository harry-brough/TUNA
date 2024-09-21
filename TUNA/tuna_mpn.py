import numpy as np
import sys

def transform_ao_two_electron_integrals(ao_two_electron_integrals, occ_mos, virt_mos, silent=False):

    """Requires two-electron integrals in atomic orbital basis, and occupied and virtual molecular orbitals.
       Uses optimised numpy.einsum to transform atomic orbital basis integrals into spin orbital basis, in four
       N^5 scaling steps, in 'ijab' shape. Returns two-electron integrals in spin orbital basis."""
    
    if not silent: print("  Transforming two-electron integrals...     ", end=""); sys.stdout.flush()

    #Using optimize in np.einsum to go via four N^5 transformations instead of N^8 transformation, generates shape 'ijab'
    mo_two_electron_integrals = np.einsum("mi,na,mnkl,kj,lb->ijab", occ_mos, virt_mos, ao_two_electron_integrals, occ_mos, virt_mos, optimize=True)

    if not silent: print("[Done]")

    return mo_two_electron_integrals


def calculate_mp2_energy_and_density(occ_epsilons, virt_epsilons, mo_two_electron_integrals, P_HF, silent=False, terse=False):

    """Requires occupied and virtual Hartree-Fock eigenvalues, two-electron spin orbital basis integrals and the Hartree-Fock 
       density matrix. Builds tensors for MP2 energy and density calculation, including MP2 wavefunction amplitudes (t), 
       coefficients (l) and four-index epsilon tensor (e_denom). Contracts tensors to form MP2 energy and density, and adds 
       occupied-occupied and virtual-virtual blocks to Hartree-Fock density matrix before symmetrising. Returns MP2 energy and 
       density matrix."""

    if not silent: print("  Calculating MP2 correlation energy...      ", end=""); sys.stdout.flush()

    n_vir = len(virt_epsilons)
    n_doubly_occ = len(occ_epsilons)


    #Setting up reciprocal four-index epsilon tensor in correct shape
    e_denom = 1 / (virt_epsilons.reshape(1, 1, n_vir, 1)  + virt_epsilons.reshape(1, 1, 1, n_vir) - occ_epsilons.reshape(n_doubly_occ, 1, 1, 1) - occ_epsilons.reshape(1, n_doubly_occ, 1, 1))

    #Setting up arrays for energy and density with correct shapes
    l = -2 * (2 * mo_two_electron_integrals - mo_two_electron_integrals.swapaxes(2,3)) * e_denom #ijab

    t = -1 * (mo_two_electron_integrals * e_denom).swapaxes(0,2).swapaxes(1,3) #abij

    #Tensor contraction for MP2 energy
    E_MP2 = np.einsum("ijab,ijab->", mo_two_electron_integrals, l / 2)
    
    if not silent: print(f"[Done]\n\n  MP2 correlation energy: {E_MP2:.10f}")

    if not silent and not terse: print("\n  Calculating MP2 unrelaxed density...       ", end=""); sys.stdout.flush()

    #Initialise MP2 density matrix as HF density matrix
    P_MP2 = P_HF

    #Tensor contraction to form occupied and virtual blocks
    P_MP2_occ = -1 * np.einsum('kiab,abkj->ij', l, t, optimize=True) 
    P_MP2_vir =  np.einsum('ijbc,acij->ab', l, t, optimize=True) 

    #Add occupied-occupied and virtual-virtual blocks to density matrix
    P_MP2[:n_doubly_occ, :n_doubly_occ] += P_MP2_occ
    P_MP2[n_doubly_occ:, n_doubly_occ:] += P_MP2_vir

    #Symmetrise matrix
    P_MP2 = (P_MP2 + P_MP2.T) / 2

    if not silent and not terse: print("[Done]\n")

    return E_MP2, P_MP2
    




def spin_component_scaling(E_MP2_SS, E_MP2_OS, silent=False):
    
    """Requires same-spin and opposite-spin MP2 energy and density components. Uses fixed scaling parameters
    to scale each component, and add them together to return scaled energy and density."""

    #Grimme's original proposed scaling factors
    same_spin_scaling = 1 / 3
    opposite_spin_scaling = 6 / 5
    
    #Scaling energy components
    E_MP2_SS_scaled = same_spin_scaling * E_MP2_SS 
    E_MP2_OS_scaled = opposite_spin_scaling * E_MP2_OS 
    
    #Forming scaled total energy
    E_MP2_scaled = E_MP2_SS_scaled + E_MP2_OS_scaled
    
    if not silent:

        print(f"[Done]\n\n  Same-spin scaling: {same_spin_scaling:.3f}")
        print(f"  Opposite-spin scaling: {opposite_spin_scaling:.3f}")
        
    return E_MP2_SS_scaled, E_MP2_OS_scaled, E_MP2_scaled
    




def calculate_scs_mp2_energy_and_density(occ_epsilons, virt_epsilons, mo_two_electron_integrals, P_HF, silent=False, terse=False):

    if not silent: print("  Calculating SCS-MP2 correlation energy...  ", end=""); sys.stdout.flush()

    n_vir = len(virt_epsilons)
    n_doubly_occ = len(occ_epsilons)

    #Setting up reciprocal four-index epsilon tensor in correct shape
    e_denom = 1 / (virt_epsilons.reshape(1, 1, n_vir, 1)  + virt_epsilons.reshape(1, 1, 1, n_vir) - occ_epsilons.reshape(n_doubly_occ, 1, 1, 1) - occ_epsilons.reshape(1, n_doubly_occ, 1, 1))

    #Setting up arrays for energy and density with correct shapes
    l = -2 * (2 * mo_two_electron_integrals - mo_two_electron_integrals.swapaxes(2,3)) * e_denom #ijab
    t = -1 * (mo_two_electron_integrals * e_denom).swapaxes(0,2).swapaxes(1,3) #abij

    #Tensor contraction for spin components of MP2 energy
    E_MP2_OS = np.einsum("ijab,abij->", mo_two_electron_integrals, t)
    E_MP2_SS = np.einsum("ijab,abij->", mo_two_electron_integrals - mo_two_electron_integrals.swapaxes(2, 3), t)

    #Scales MP2 energy spin components
    E_MP2_SS_scaled, E_MP2_OS_scaled, E_MP2_scaled = spin_component_scaling(E_MP2_SS, E_MP2_OS, silent)

    if not silent: 

        print(f"\n  Same-spin-scaled energy: {E_MP2_SS_scaled:.10f}")
        print(f"  Opposite-spin-scaled energy: {E_MP2_OS_scaled:.10f}")

    if not silent: print(f"\n  SCS-MP2 correlation energy: {E_MP2_scaled:.10f}")

    if not silent and not terse: print("\n  Calculating MP2 unrelaxed density...       ", end=""); sys.stdout.flush()

    #Initialise MP2 density matrix as HF density matrix
    P_MP2 = P_HF


    #Tensor contraction to form occupied and virtual blocks
    P_MP2_occ = -1 * np.einsum('kiab,abkj->ij', l, t, optimize=True) 
    P_MP2_vir =  np.einsum('ijbc,acij->ab', l, t, optimize=True) 

    #Add occupied-occupied and virtual-virtual blocks to density matrix
    P_MP2[:n_doubly_occ, :n_doubly_occ] += P_MP2_occ
    P_MP2[n_doubly_occ:, n_doubly_occ:] += P_MP2_vir

    #Symmetrise matrix
    P_MP2 = (P_MP2 + P_MP2.T) / 2

    if not silent and not terse: print("[Done]")



    return  E_MP2_scaled, P_MP2
    


def spin_block_two_electron_integrals(V_EE_ao,  molecular_orbitals_alpha, molecular_orbitals_beta, silent=False):

    if not silent: print("  Spin-blocking two-electron integrals...    ", end=""); sys.stdout.flush()

    C_spin_block = np.block([[molecular_orbitals_alpha, np.zeros_like(molecular_orbitals_beta)], [np.zeros_like(molecular_orbitals_alpha), molecular_orbitals_beta]])

    V_EE_spin_block = np.kron(np.eye(2), np.kron(np.eye(2), V_EE_ao).T)

    V_EE_spin_block = V_EE_spin_block - V_EE_spin_block.transpose(0, 2, 1, 3)

    if not silent: print("[Done]")
    
    return V_EE_spin_block, C_spin_block



def transform_spin_blocked_two_electron_integrals(V_EE_spin_block, C_spin_block, silent=False):
    
    if not silent: print("  Transforming two-electron integrals...     ", end=""); sys.stdout.flush()

    so_two_electron_integrals = np.einsum("mi,nj,mkln,ka,lb->ijab", C_spin_block, C_spin_block, V_EE_spin_block, C_spin_block, C_spin_block, optimize=True)

    if not silent: print("[Done]\n")

    return so_two_electron_integrals



def calculate_spin_orbital_MP2_energy(e_ijab, so_two_electron_integrals, o, v, silent=False):


    if not silent: print("  Calculating MP2 correlation energy...      ", end=""); sys.stdout.flush()

    E_MP2 = (1 / 4) * np.einsum('ijab,abij,ijab->', so_two_electron_integrals[o, o, v, v], so_two_electron_integrals[v, v, o, o], e_ijab, optimize=True)

    if not silent: print(f"[Done]\n\n  MP2 correlation energy: {E_MP2:.10f}\n")

    return E_MP2



def calculate_spin_orbital_MP3_energy(e_ijab, so_two_electron_integrals, o, v, silent=False):

    if not silent: print("  Calculating MP3 correlation energy...      ", end=""); sys.stdout.flush()
        
    E_MP3 = (1 / 8) * np.einsum('ijab,klij,abkl,ijab,klab->', so_two_electron_integrals[o, o, v, v], so_two_electron_integrals[o, o, o, o], so_two_electron_integrals[v, v, o, o], e_ijab, e_ijab, optimize=True)
    E_MP3 += (1 / 8) * np.einsum('ijab,abcd,cdij,ijab,ijcd->', so_two_electron_integrals[o, o, v, v], so_two_electron_integrals[v, v, v, v], so_two_electron_integrals[v, v, o, o], e_ijab, e_ijab, optimize=True)
    E_MP3 += np.einsum('ijab,kbcj,acik,ijab,ikac->', so_two_electron_integrals[o, o, v, v], so_two_electron_integrals[o, v, v, o], so_two_electron_integrals[v, v, o, o], e_ijab, e_ijab, optimize=True)

    if not silent: print(f"[Done]\n\n  MP3 correlation energy: {E_MP3:.10f}")

    return E_MP3



def build_epsilons_tensor(epsilons_sorted, o, v):

    n = np.newaxis

    e_ijab = 1 / (epsilons_sorted[o, n, n, n] + epsilons_sorted[n, o, n, n] - epsilons_sorted[n, n, v, n] - epsilons_sorted[n, n, n, v])

    return e_ijab