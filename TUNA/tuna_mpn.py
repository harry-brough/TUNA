import numpy as np
import sys
from tuna_util import *


def transform_ao_two_electron_integrals(ao_two_electron_integrals, occ_mos, virt_mos, calculation, silent=False):

    """
    
    Requires two-electron integrals in atomic orbital basis (array), occupied and virtual sets of molecular orbitals (arrays) and calculation (Calculation).

    Uses optimised numpy einsum to transform AO basis integrals into molecular orbtial basis integrals, in four N^5 scaling steps, in "ijab" shape.

    Returns molecular orbital basis two-electron integrals.

    """

    #Doesn't print during gradient calculations or MD simulations
    if not silent: log("  Transforming two-electron integrals...     ", calculation, 1, end=""); sys.stdout.flush()

    #Using optimize in np.einsum to go via four N^5 transformations instead of N^8 transformation, generates shape 'ijab'
    mo_two_electron_integrals = np.einsum("mi,na,mnkl,kj,lb->ijab", occ_mos, virt_mos, ao_two_electron_integrals, occ_mos, virt_mos, optimize=True)

    if not silent: log("[Done]", calculation, 1)

    return mo_two_electron_integrals



def calculate_mp2_energy_and_density(occ_epsilons, virt_epsilons, mo_two_electron_integrals, P_HF, calculation, silent=False, terse=False):

    """
    
    Requires occupied and virtual Hartree-Fock eigenvalues (arrays), molecular orbital basis two-electron integrals (array), Hartree-Fock density matrix (array)
    and calculation (Calculation).

    Builds tensors for MP2 energy and density calculation, including MP2 wavefunction ammplitudes (t) and coefficients (l) and four-index epsilon tensor. Contracts
    tensors to form the MP2 energy and density, and adds occupied-occupied and virtual-virtual blocks to the HF density matrix before symmetrising.

    Returns MP2 energy and unrelaxed MP2 density matrix.
    
    """

    if not silent: log("  Calculating MP2 correlation energy...      ", calculation, 1, end=""); sys.stdout.flush()

    n_vir = len(virt_epsilons)
    n_doubly_occ = len(occ_epsilons)


    #Setting up reciprocal four-index epsilon tensor in correct shape
    e_denom = 1 / (virt_epsilons.reshape(1, 1, n_vir, 1)  + virt_epsilons.reshape(1, 1, 1, n_vir) - occ_epsilons.reshape(n_doubly_occ, 1, 1, 1) - occ_epsilons.reshape(1, n_doubly_occ, 1, 1))

    #Setting up arrays for energy and density with correct shapes
    l = -2 * (2 * mo_two_electron_integrals - mo_two_electron_integrals.swapaxes(2,3)) * e_denom #ijab
    t = -1 * (mo_two_electron_integrals * e_denom).swapaxes(0,2).swapaxes(1,3) #abij

    #Tensor contraction for MP2 energy
    E_MP2 = np.einsum("ijab,ijab->", mo_two_electron_integrals, l / 2)
    
    if not silent: log(f"[Done]\n\n  MP2 correlation energy: {E_MP2:.10f}", calculation, 1)

    if not silent and not terse: log("\n  Calculating MP2 unrelaxed density...       ", calculation, 2, end=""); sys.stdout.flush()

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

    if not silent and not terse: log("[Done]", calculation, 2)

    return E_MP2, P_MP2
    




def spin_component_scaling(E_MP2_SS, E_MP2_OS, calculation, silent=False):
    
    """
    
    Requires same-spin and opposite-spin MP2 energy and density components. 
    
    Uses fixed scaling parameters to scale each component, and add them together.
    
    Returns same-spin scaled and opposite-spin scaled and total scaled MP2 energies (floats).
    
    """

    #Grimme's original proposed scaling factors
    same_spin_scaling = 1 / 3
    opposite_spin_scaling = 6 / 5
    
    #Scaling energy components
    E_MP2_SS_scaled = same_spin_scaling * E_MP2_SS 
    E_MP2_OS_scaled = opposite_spin_scaling * E_MP2_OS 
    
    #Forming scaled total energy
    E_MP2_scaled = E_MP2_SS_scaled + E_MP2_OS_scaled
    
    if not silent:

        log(f"[Done]\n\n  Same-spin scaling: {same_spin_scaling:.3f}", calculation, 1)
        log(f"  Opposite-spin scaling: {opposite_spin_scaling:.3f}", calculation, 1)
        

    return E_MP2_SS_scaled, E_MP2_OS_scaled, E_MP2_scaled
    




def calculate_scs_mp2_energy_and_density(occ_epsilons, virt_epsilons, mo_two_electron_integrals, P_HF, calculation, silent=False, terse=False):

    """
    
    Requires occupied and virtual Hartree-Fock eigenvalues (arrays), molecular orbital basis two-electron integrals (array), Hartree-Fock density matrix (array)
    and calculation (Calculation).

    Builds tensors for MP2 energy and density calculation, including MP2 wavefunction ammplitudes (t) and coefficients (l) and four-index epsilon tensor. Contracts
    tensors to form the same-spin and opposite contributions to MP2 energy. Also creates the unscaled MP2 density matrix.

    Returns spin-component-scaled MP2 energy and unrelaxed unscaled MP2 density matrix.
    
    """

    if not silent: log("  Calculating SCS-MP2 correlation energy...  ", calculation, 1, end=""); sys.stdout.flush()

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
    E_MP2_SS_scaled, E_MP2_OS_scaled, E_MP2_scaled = spin_component_scaling(E_MP2_SS, E_MP2_OS, calculation, silent)

    if not silent: 

        log(f"\n  Same-spin-scaled energy: {E_MP2_SS_scaled:.10f}", calculation, 1)
        log(f"  Opposite-spin-scaled energy: {E_MP2_OS_scaled:.10f}", calculation, 1)

    if not silent: log(f"\n  SCS-MP2 correlation energy: {E_MP2_scaled:.10f}", calculation, 1)

    if not silent and not terse: log("\n  Calculating MP2 unrelaxed density...       ", calculation, 1, end=""); sys.stdout.flush()

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

    if not silent and not terse: log("[Done]", calculation, 1)


    return  E_MP2_scaled, P_MP2
    



def spin_block_two_electron_integrals(V_EE_ao, molecular_orbitals_alpha, molecular_orbitals_beta, calculation, silent=False):

    """
    
    Requires atomic orbital basis two-electron integrals (array), alpha and beta molecular orbitals (arrays) and calculation (Calculation).

    Spin-blocks two-electron integrals and molecular orbitals, separating alpha and beta spins in the array.
    
    Returns spin-blocked two-electron integrals (array) and molecular orbitals (array).

    """

    if not silent: log("  Spin-blocking two-electron integrals...    ", calculation, 2, end=""); sys.stdout.flush()

    #Joins alpha and beta orbitals into one big array
    C_spin_block = np.block([[molecular_orbitals_alpha, np.zeros_like(molecular_orbitals_beta)], [np.zeros_like(molecular_orbitals_alpha), molecular_orbitals_beta]])

    V_EE_spin_block = np.kron(np.eye(2), np.kron(np.eye(2), V_EE_ao).T)

    #Antisymmetrises spin-blocked two-electron integrals
    V_EE_spin_block = V_EE_spin_block - V_EE_spin_block.transpose(0, 2, 1, 3)

    if not silent: log("[Done]", calculation, 2)
    
    return V_EE_spin_block, C_spin_block



def transform_spin_blocked_two_electron_integrals(V_EE_spin_block, C_spin_block, calculation, silent=False):

    """
    
    Requires spin-blocked two-electron integrals (array), spin-blocked molecular orbitals (array) and calculation (Calculation).

    Transforms spin-blocked two-electron integrals into spin orbital basis by N^5 steps. 

    Returns spin orbital basis two-electron integrals.

    """ 

    #Keeps silent if in a gradient calculation or MD simulation, for instance 
    if not silent: log("  Transforming two-electron integrals...     ", calculation, 1, end=""); sys.stdout.flush()

    so_two_electron_integrals = np.einsum("mi,nj,mkln,ka,lb->ijab", C_spin_block, C_spin_block, V_EE_spin_block, C_spin_block, C_spin_block, optimize=True)

    if not silent: log("[Done]\n", calculation, 1)

    return so_two_electron_integrals



def calculate_spin_orbital_MP2_energy(e_ijab, so_two_electron_integrals, o, v, calculation, silent=False):

    """
    
    Requires epsilons tensor (array), spin orbital basis two-electron integrals (array), occupied and virtual slices (slices), and calculation (Calculation).
    
    Calculates the MP2 contribution to energy, by numpy einsum optimised tensor contraction in N^5 step.
    
    Returns the MP2 energy.
    
    """

    #Keeps silent if in a gradient calculation or MD simulation, for instance
    if not silent: log("  Calculating MP2 correlation energy...      ", calculation, 1, end=""); sys.stdout.flush()

    E_MP2 = (1 / 4) * np.einsum('ijab,abij,ijab->', so_two_electron_integrals[o, o, v, v], so_two_electron_integrals[v, v, o, o], e_ijab, optimize=True)

    if not silent: log(f"[Done]\n\n  MP2 correlation energy: {E_MP2:.10f}", calculation, 1)

    return E_MP2



def calculate_spin_orbital_MP3_energy(e_ijab, so_two_electron_integrals, o, v, calculation, silent=False):

    """
    
    Requires epsilons tensor (array), spin orbital basis two-electron integrals (array), occupied and virtual slices (slices) and calculation (Calculation).

    Calculates the MP3 contribution to energy, by several numpy einsum optimised tensor contractions, with N^7 scaling.

    Returns the MP3 energy.

    """

    #Keeps silent if in a gradient calculation or MD simulation, for instance
    if not silent: log("\n  Calculating MP3 correlation energy...      ", calculation, 1, end=""); sys.stdout.flush()
        
    E_MP3 = (1 / 8) * np.einsum('ijab,klij,abkl,ijab,klab->', so_two_electron_integrals[o, o, v, v], so_two_electron_integrals[o, o, o, o], so_two_electron_integrals[v, v, o, o], e_ijab, e_ijab, optimize=True)
    E_MP3 += (1 / 8) * np.einsum('ijab,abcd,cdij,ijab,ijcd->', so_two_electron_integrals[o, o, v, v], so_two_electron_integrals[v, v, v, v], so_two_electron_integrals[v, v, o, o], e_ijab, e_ijab, optimize=True)
    E_MP3 += np.einsum('ijab,kbcj,acik,ijab,ikac->', so_two_electron_integrals[o, o, v, v], so_two_electron_integrals[o, v, v, o], so_two_electron_integrals[v, v, o, o], e_ijab, e_ijab, optimize=True)

    if not silent: log(f"[Done]\n\n  MP3 correlation energy: {E_MP3:.10f}", calculation, 1)

    return E_MP3



def build_epsilons_tensor(epsilons_sorted, o, v):

    """
    
    Requires sorted epsilons (array), occupied and virtual slices (slices).
    
    Calculates and returns four dimensional epsilons tensor, indexed "ijab".
    
    """

    n = np.newaxis

    e_ijab = 1 / (epsilons_sorted[o, n, n, n] + epsilons_sorted[n, o, n, n] - epsilons_sorted[n, n, v, n] - epsilons_sorted[n, n, n, v])

    return e_ijab





def calculate_moller_plesset(mp2_type, method, molecule, scf_output, V_EE_ao_basis, calculation, silent=False, terse=False):

    """
    
    Requires MP2 type (molecular or spin orbital basis, string), method (string), molecule (Molecule), energy output (Output), atomic orbital basis two-electron
    integrals (array) and calculation (Calculation).

    If a molecular orbital basis calculation is requested, the MP2 energy and density are requested, optionally including spin-component scaling. If a spin orbital 
    basis calculation is requested, only the energy is calculated, which may be MP3 or MP2. If a SCS-MP3 calculation is requested, the MO calculations happen first,
    before the SO calculations.

    Returns MP2 energy (float) and MP3 energy (float).

    """


    S = scf_output.S
    P = scf_output.P

    E_MP2 = 0
    E_MP3 = 0

    #For RMP2, SCS-MP2
    if mp2_type == "MO":

        #Unpacks useful quantities
        n_doubly_occ = molecule.n_doubly_occ
        epsilons = scf_output.epsilons
        molecular_orbitals = scf_output.molecular_orbitals
    
        if not silent:

            log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
            log("         MP2 Energy and Density Calculation ", calculation, 1)
            log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)

        #Transforms density matrix into molecular orbital basis, from atomic orbital basis density matrix
        P_HF_mo_basis = molecular_orbitals.T @ S @ P @ S @ molecular_orbitals

        #Slices up occupied and virtual eigenvectors and eigenvalues
        occupied_mos = molecular_orbitals[:, :n_doubly_occ]
        virtual_mos = molecular_orbitals[:, n_doubly_occ:]
        
        occupied_epsilons = epsilons[:n_doubly_occ]
        virtual_epsilons = epsilons[n_doubly_occ:]

        #Transforms AO basis two-electron integrals into the MO basis
        V_EE_mo_basis = transform_ao_two_electron_integrals(V_EE_ao_basis, occupied_mos, virtual_mos, calculation, silent=silent)


        if method == "MP2": E_MP2, P_MP2_mo_basis = calculate_mp2_energy_and_density(occupied_epsilons, virtual_epsilons, V_EE_mo_basis, P_HF_mo_basis, calculation, silent=silent, terse=terse)
        
        elif method in ["SCS-MP2", "SCS-MP3"]: 
            
            E_MP2, P_MP2_mo_basis = calculate_scs_mp2_energy_and_density(occupied_epsilons, virtual_epsilons, V_EE_mo_basis, P_HF_mo_basis, calculation, silent=silent, terse=terse)
        
            if not silent: warning("Density is not spin-component-scaled!", 2)

        #Back transforms MO basis MP2 unrelaxed density matrix to AO basis
        P_MP2 = molecular_orbitals @ P_MP2_mo_basis @ molecular_orbitals.T

        #Diagonalises MO basis into natural orbitals, and calculates sum of eigenvalues (natural orbital occupancies) 
        natural_orbital_occupancies = np.sort(np.linalg.eigh(P_MP2_mo_basis)[0])[::-1]
        sum_of_occupancies = np.sum(natural_orbital_occupancies)
        

        if not silent: 
            
            log("\n  Natural orbital occupancies: \n", calculation, 2)

            #Prints out all the natural orbital occupancies, the sum and the trace of the density matrix
            for i in range(len(natural_orbital_occupancies)): log(f"    {i + 1}.   {natural_orbital_occupancies[i]:.10f}", calculation, 2)

            log(f"\n  Sum of natural orbital occupancies: {sum_of_occupancies:.6f}", calculation, 2)
            log(f"  Trace of density matrix:  {np.trace(P_MP2_mo_basis):.6f}", calculation, 2)

        if not silent: log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)

        #For SCS-MP3, the SO calculation is also needed
        if method == "SCS-MP3": 
            
            mp2_type = "SO"
            scs_mp2_energy = E_MP2

        elif terse: log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)

        P = P_MP2

    #For UMP2, MP3, SCS-MP3
    if mp2_type == "SO":

        molecular_orbitals_alpha = scf_output.molecular_orbitals_alpha

        #Handles the triplet case, with no beta electrons
        if molecule.n_beta == 0: 
            
            epsilons_combined = scf_output.epsilons_alpha
            molecular_orbitals_beta = molecular_orbitals_alpha

        else: 
            
            epsilons_combined = scf_output.epsilons_combined
            molecular_orbitals_beta = scf_output.molecular_orbitals_beta


        n_occ = molecule.n_occ


        if not silent:

            log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
            if method in ["MP2", "UMP2"]: log("                MP2 Energy Calculation ", calculation, 1)
            elif method in ["MP3", "UMP3", "SCS-MP3"]: log("                MP3 Energy Calculation ", calculation, 1)

            log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
 
        
        #Spin-blocks two-electron integrals and molecular orbitals
        V_EE_spin_block, C_spin_block = spin_block_two_electron_integrals(V_EE_ao_basis, molecular_orbitals_alpha, molecular_orbitals_beta, calculation, silent=silent)

        #Reorganises spin-blocked molecular orbitals
        C_spin_block = C_spin_block[:, epsilons_combined.argsort()] 

        #Sorts epsilons from lowest to highest value
        epsilons_sorted = np.sort(epsilons_combined)

        #Transforms spin-blocked two-electron integrals into spin-orbital basis
        so_two_electron_integrals = transform_spin_blocked_two_electron_integrals(V_EE_spin_block, C_spin_block, calculation, silent=silent)

        #Defines occupied and virtual slices
        o = slice(None, n_occ)
        v = slice(n_occ, None)

        #Builds four-dimensional epsilons tensor
        e_ijab = build_epsilons_tensor(epsilons_sorted, o, v)

        #Calculates MP2 energy from spin-orbital basis
        if method != "SCS-MP3": E_MP2 = calculate_spin_orbital_MP2_energy(e_ijab, so_two_electron_integrals, o, v, calculation, silent=silent)

        #Calculates MP3 energy from spin-orbital basis
        if method in ["MP3", "UMP3", "SCS-MP3"]: E_MP3 = calculate_spin_orbital_MP3_energy(e_ijab, so_two_electron_integrals, o, v, calculation, silent=silent)

        #Applies Grimme's default scaling to MP3 energy if SCS-MP3 is requested, then prints this information
        if method == "SCS-MP3":

            mp3_scaling = 1 / 4
            E_MP3 *= mp3_scaling
            
            if not silent: 
                
                log(f"  Scaling for MP3: {mp3_scaling}\n", calculation, 1)
                log(f"  Scaled MP3 correlation energy: {E_MP3:.10f}", calculation, 1)
                log(f"  SCS-MP3 correlation energy: {(E_MP3 + scs_mp2_energy):.10f}", calculation, 1)
        

        if not silent: log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)


    return E_MP2, E_MP3, P
