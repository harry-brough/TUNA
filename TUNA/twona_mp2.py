import numpy as np
import sys
import tuna_util as util


def build_unrestricted_molecular_orbital_matrix(molecular_orbitals_alpha, molecular_orbitals_beta):

    molecular_orbitals = np.block([[molecular_orbitals_alpha, np.zeros_like(molecular_orbitals_beta)],
                                    [np.zeros_like(molecular_orbitals_alpha), molecular_orbitals_beta]])

    return molecular_orbitals



def spin_block_mo_two_electron_integrals(mo_two_electron_integrals):
    """  

    identity = np.eye(2)
    Imo_two_electron_integrals = np.kron(identity, Imo_two_electron_integrals)

    return np.kron(identity, mo_two_electron_integrals.T)



def transform_ao_two_electron_integrals(ao_two_electron_integrals, occ_mos, virt_mos, silent=False):

    """Requires two-electron integrals in atomic orbital basis, and occupied and virtual molecular orbitals.
       Uses optimised numpy.einsum to transform atomic orbital basis integrals into spin orbital basis, in four
       N^5 scaling steps, in 'ijab' shape. Returns two-electron integrals in spin orbital basis."""
    
    if not silent: print("  Transforming two-electron integrals...     ", end=""); sys.stdout.flush()

    #Using optimize in np.einsum to go via four N^5 transformations instead of N^8 transformation, generates shape 'ijab'
    mo_two_electron_integrals = np.einsum("mp,nq,mnkl,kr,ls->prqs", occ_mos, virt_mos, ao_two_electron_integrals, occ_mos, virt_mos, optimize=True)

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

    if not terse:

        #Tensor contraction to form occupied and virtual blocks
        P_MP2_occ = -1 * np.einsum('kiab,abkj->ij', l, t, optimize=True) 
        P_MP2_vir =  np.einsum('ijbc,acij->ab', l, t, optimize=True) 

        #Add occupied-occupied and virtual-virtual blocks to density matrix
        P_MP2[:n_doubly_occ, :n_doubly_occ] += P_MP2_occ
        P_MP2[n_doubly_occ:, n_doubly_occ:] += P_MP2_vir

        #Symmetrise matrix
        P_MP2 = (P_MP2 + P_MP2.T) / 2

    if not silent and not terse: print("[Done]")

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
        print(f"[Done]\n\n  SCS-MP2 correlation energy: {E_MP2_scaled:.10f}\n")
        print(f"  Same-spin scaling: {same_spin_scaling:.3f}")
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


    if not silent and not terse: print("\n  Calculating MP2 unrelaxed density...       ", end=""); sys.stdout.flush()
    
    #Initialise MP2 density matrix as HF density matrix
    P_MP2 = P_HF

    if not terse:

        #Tensor contraction to form occupied and virtual blocks
        P_MP2_occ = -1 * np.einsum('kiab,abkj->ij', l, t, optimize=True) 
        P_MP2_vir =  np.einsum('ijbc,acij->ab', l, t, optimize=True) 

        #Add occupied-occupied and virtual-virtual blocks to density matrix
        P_MP2[:n_doubly_occ, :n_doubly_occ] += P_MP2_occ
        P_MP2[n_doubly_occ:, n_doubly_occ:] += P_MP2_vir

        #Symmetrise matrix
        P_MP2 = (P_MP2 + P_MP2.T) / 2

    if not silent and not terse: 
        
        print("[Done]")
        util.warning("This is the unscaled MP2 unrelaxed density!")

    """WARNING: This is the unscaled MP2 density!"""

    return  E_MP2_scaled, P_MP2
    
