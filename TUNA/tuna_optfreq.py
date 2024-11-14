from tuna_util import *
import numpy as np
import tuna_energy as energ
import sys
from termcolor import colored
import tuna_postscf as postscf
          


def calculate_gradient(coordinates, calculation, atoms, silent=False):

    """
    
    Requires coordinates (array), calculation (Calculation), and atoms (list).

    Prods coordinates backwards and forwards, then calculates energy at both points to calculate an accurate central differences derivative.

    Returns the bond length derivative of the energy.
    
    """

    #Chosen to maintain numerical stability while giving accurate derivatives
    prod = 0.0001

    prodding_coords = np.array([[0,0,0], [0,0, prod]])  
    forward_coords = coordinates + prodding_coords
    backward_coords = coordinates - prodding_coords

    if not silent: log(" Calculating energy on displaced geometry 1 of 2...  ", calculation, 1, end=""); sys.stdout.flush()
    _, _, forward_energy, _ = energ.calculate_energy(calculation, atoms, forward_coords, silent=True)
    if not silent: log("[Done]", calculation, 1)

    if not silent: log(" Calculating energy on displaced geometry 2 of 2...  ", calculation, 1, end=""); sys.stdout.flush()
    _, _, backward_energy, _ = energ.calculate_energy(calculation, atoms, backward_coords, silent=True)
    if not silent: log("[Done]", calculation, 1)
    
    gradient = (forward_energy - backward_energy) / (2 * prod)

    return gradient
    


def calculate_approximate_hessian(delta_x, delta_grad): 

    """
    
    Requires change in bond length (float) and change in gradient (float).

    Calculates approximate Hessian by finding the quotient.

    Returns approximate Hessian (float).
    
    """

    hessian = delta_grad / delta_x

    return hessian



def calculate_hessian(coordinates, calculation, atoms, silent=False):

    """
    
    Requires coordinates (array), calculation (Calculation), and atoms (list).

    Prods coordinates backwards and forwards twice, then calculates energy at all points to calculate an accurate central differences second derivative.

    Returns the second bond length derivative of the energy.
    
    """

    #Chosen to maintain numerical stability while giving accurate derivatives
    prod = 0.0001

    prodding_coords = np.array([[0,0,0], [0,0, prod]])  

    far_forward_coords = coordinates + 2 * prodding_coords
    forward_coords = coordinates + prodding_coords
    backward_coords = coordinates - prodding_coords
    far_backward_coords = coordinates - 2 * prodding_coords  

    if not silent: log("\n Calculating energy on displaced geometry 1 of 5...  ", calculation, 1, end=""); sys.stdout.flush()
    _, _, far_forward_energy, _ = energ.calculate_energy(calculation, atoms, far_forward_coords, silent=True)
    if not silent: log("[Done]", calculation, 1)   

    if not silent: log(" Calculating energy on displaced geometry 2 of 5...  ", calculation, 1, end=""); sys.stdout.flush()
    _, _, forward_energy, _ = energ.calculate_energy(calculation, atoms, forward_coords, silent=True)
    if not silent: log("[Done]", calculation, 1)   

    if not silent: log(" Calculating energy on displaced geometry 3 of 5...  ", calculation, 1, end=""); sys.stdout.flush()
    _, _, energy, _ = energ.calculate_energy(calculation, atoms, coordinates, silent=True)
    if not silent: log("[Done]", calculation, 1)   

    if not silent: log(" Calculating energy on displaced geometry 4 of 5...  ", calculation, 1, end=""); sys.stdout.flush()
    _, _, backward_energy, _ = energ.calculate_energy(calculation, atoms, backward_coords, silent=True)
    if not silent: log("[Done]", calculation, 1)   

    if not silent: log(" Calculating energy on displaced geometry 5 of 5...  ", calculation, 1, end=""); sys.stdout.flush()
    _, _, far_backward_energy, _  = energ.calculate_energy(calculation, atoms, far_backward_coords, silent=True)
    if not silent: log("[Done]\n", calculation, 1)   

    #Equation from Wikipedia page on numerical second derivative methods, fairly noise-resistant formula
    hessian = (-far_forward_energy + 16 * forward_energy - 30 * energy + 16 * backward_energy - far_backward_energy) / (12 * prod ** 2)

    return hessian



def optimise_geometry(calculation, atoms, coordinates):
    
    """
    
    Requires calculation (Calculation), atoms (list) and starting coordinates (array).

    Prints user-defined geometry optimisation options, then calculates the gradient at each step, uses the exact or approximate
    Hessian to update the new coordinates and repeats this process until the change in gradient and step are below the convergence
    criteria defined.

    Returns optimised molecule (Molecule) with final energy (float) if optimisation completes successfully.
    
    """
    
    #Unpacks useful quantities
    maximum_step = angstrom_to_bohr(calculation.max_step)
    default_hessian = calculation.default_hessian
    geom_conv_criteria = calculation.geom_conv
    max_geom_iter = calculation.geom_max_iter
    optmax = calculation.optmax


    log("\nInitialising geometry optimisation...\n", calculation, 1)

    #If TRAJ keyword is used, prints trajectory to file
    if calculation.trajectory: 
        
        log("Printing trajectory data to \"tuna-trajectory.xyz\"\n", calculation, 1)
        
        with open('tuna-trajectory.xyz', 'w'): pass

    if not calculation.calchess: log(f"Using approximate Hessian in convex region, Hessian of {default_hessian:.3f} outside.\n", calculation, 1)
    else: log(f"Using exact Hessian in convex region, Hessian of {default_hessian:.3f} outside.\n", calculation, 1)

    log(f"Gradient convergence: {geom_conv_criteria.get("gradient"):.7f}", calculation, 1)
    log(f"Step convergence: {geom_conv_criteria.get("step"):.7f}", calculation, 1)
    log(f"Maximum iterations: {max_geom_iter}", calculation, 1)
    log(f"Maximum step: {bohr_to_angstrom(maximum_step):.5f}", calculation, 1)

    P_guess = None; E_guess = 0; P_guess_alpha = None; P_guess_beta = None

    for iteration in range(1, max_geom_iter + 1):

        log(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        log(f"Beginning energy and gradient calculation on geometry iteration number {iteration}...", calculation, 1)
        log(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        
        if not calculation.moread: 

            P_guess = None
            P_guess_alpha = None
            P_guess_beta = None
            E_guess = 0

        if calculation.additional_print: scf_output, molecule, energy, final_P = energ.calculate_energy(calculation, atoms, coordinates, P_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, E_guess=E_guess, terse=False)
        else: scf_output, molecule, energy, final_P = energ.calculate_energy(calculation, atoms, coordinates, P_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, E_guess=E_guess, terse=True)

        #Sets density matrix guess to the last density matrix of each kind
        P = final_P
        P_guess =  scf_output.P
        P_guess_alpha = scf_output.P_alpha
        P_guess_beta = scf_output.P_beta

        E_guess = energy

        #Calculates gradient at each point
        log("\n Beginning numerical gradient calculation...  \n", calculation, 1)
        gradient = calculate_gradient(coordinates, calculation, atoms)

        bond_length = molecule.bond_length
        hessian = default_hessian

        if gradient > 0: space = "  "
        else: space = "  "

     
        if iteration > 1:

            if calculation.calchess: 

                log("\n Beginning calculation of exact Hessian...", calculation, 1)
                h = calculate_hessian(coordinates, calculation, atoms)

            else: 

                #Calculates approximate Hessian if CALCHESS keyword not used
                h = calculate_approximate_hessian(bond_length - old_bond_length, gradient - old_gradient)


            #Checks if region is convex or concave, if in the correct region for opt to min/max, sets the hessian to the second derivative
            if optmax:

                if h < 0.01: hessian = -h

            else: 

                if h > 0.01: hessian = h


        #Calculates step to be taken using Wikipedia equation for Newton's method
        inverse_hessian = 1 / hessian           
        step = inverse_hessian * gradient
        
        #Checks for convergence of various criteria for optimisation
        if np.abs(gradient) < geom_conv_criteria.get("gradient"): converged_grad = True; conv_check_grad = "Yes"
        else: converged_grad = False; conv_check_grad = "No"

        if np.abs(step) < geom_conv_criteria.get("step"): converged_step = True; conv_check_step = "Yes"
        else: converged_step = False; conv_check_step = "No"
        
        log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        log("   Factor       Value      Conv. Criteria    Converged", calculation, 1)
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        log(f"  Gradient    {gradient:.8f}  {space} {geom_conv_criteria.get("gradient"):.8f}   {space}   {conv_check_grad} ", calculation, 1)
        log(f"    Step      {step:.8f}  {space} {geom_conv_criteria.get("step"):.8f}   {space}   {conv_check_step} ", calculation, 1)
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)

        #Prints trajectory to file if TRAJ keyword has been used
        if calculation.trajectory: print_trajectory(molecule, energy, coordinates)

        #If optimisation is converged, begin post SCF output and print to console, then finish calculation
        if converged_grad and converged_step: 

            log("\n==========================================", calculation, 1)           
            log(colored(f" Optimisation converged in {iteration} iterations!","white"), calculation, 1)
            log("==========================================", calculation, 1)

            postscf.post_scf_output(molecule, calculation, scf_output.epsilons, scf_output.molecular_orbitals, P, scf_output.S, molecule.ao_ranges, scf_output.D, scf_output.P_alpha, scf_output.P_beta, scf_output.epsilons_alpha, scf_output.epsilons_beta, scf_output.molecular_orbitals_alpha, scf_output.molecular_orbitals_beta)
          
            log(f"\n Optimisation converged in {iteration} iterations to bond length of {bohr_to_angstrom(bond_length):.6f} angstroms!", calculation, 1)
            log(f"\n Final single point energy: {energy:.10f}", calculation, 1)

            return molecule, energy

        else:
            
            if step > maximum_step: 

                step = maximum_step
                warning("Calculated step is outside of trust radius, taking maximum step instead.")

            elif step < -maximum_step:

                step = -maximum_step
                warning("Calculated step is outside of trust radius, taking maximum step instead.")

            #Checks direction in which step should be taken, depending on whether OPTMAX keyword has been used
            direction = -1 if optmax else 1

            #Builds new coordinates
            coordinates = np.array([[0, 0, 0], [0, 0, coordinates[1][2] - direction * step]])

            if coordinates[1][2] <= 0: error("Optimisation generated negative bond length! Decrease trust radius!")

            #Updates "old" quantities to be used for comparison to check convergence
            old_bond_length = bond_length
            old_gradient = gradient
     

    error(F"Geometry optimisation did not converge in {max_geom_iter} iterations! Increase the maximum or give up!")





def calculate_frequency(calculation, atoms=None, coordinates=None, optimised_molecule=None, optimised_energy=None):

    """
    
    Requires calculation (Calculation). Needs a molecule, either given by atoms (list) and coordinates (array) or by
    an optimised molecule (Molecule) and optimised energy (float) if used as "OPTFREQ" keyword.

    Calculates the harmonic frequency of a given molecule, by determining the numerical second derivatives. Then prints out
    this frequency information, as well as thermochemical data.

    """

    #If "FREQ" keyword has been used, calculates the energy using the supplied atoms and coordinates, otherwise uses the supplied molecule and energy
    if calculation.calculation_type == "FREQ":
          
        _, molecule, energy, _ = energ.calculate_energy(calculation, atoms, coordinates)
    
    else:

        molecule = optimised_molecule
        energy = optimised_energy

    #Unpacks useful molecular quantities
    point_group = molecule.point_group
    bond_length = molecule.bond_length
    atoms = molecule.atoms
    coordinates = molecule.coordinates
    masses = molecule.masses

    #Unpacks useful calculation quantities from user-defined parameters
    temperature = calculation.temperature
    pressure = calculation.pressure  


    log("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log("Beginning TUNA harmonic frequency calculation...", calculation, 1)
    log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)

    
    log(f"\n Hessian will be calculated at a bond length of {bohr_to_angstrom(bond_length):.6f} angstroms.", calculation, 1)
    
    #Spring stiffness is calculates as the Hessian, through numerical second derivatives
    k = calculate_hessian(coordinates, calculation, atoms)

    #Reduced mass calculated in order to calculate frequency of harmonic oscillator
    reduced_mass = postscf.calculate_reduced_mass(masses)

    #Checks if an imaginary mode is present, and if so, appends an "i" and sets the vibrational entropy, internal energy and zero-point energy to zero
    if k > 0:
    
        frequency_hartree = np.sqrt(k / reduced_mass)
        i = ""
        zpe = 0.5 * frequency_hartree
        
    else:   
    
        frequency_hartree = np.sqrt(-k / reduced_mass)
        i = " i"
        zpe = 0
        vibrational_entropy = 0; 
        vibrational_internal_energy = 0
        
        warning("Imaginary frequency calculated! Zero-point energy and vibrational thermochemistry set to zero!\n")

    #Converts frequency into human units from atomic units
    frequency_per_cm = frequency_hartree * constants.per_cm_in_hartree
    

    log(" Using masses of most abundant isotopes...\n", calculation, 1)

    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log("       Harmonic Frequency", calculation, 1)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log(f"  Force constant: {k:.5f}", calculation, 1)
    log(f"  Reduced mass: {reduced_mass:.2f}", calculation, 1)
    log(f"\n  Frequency (per cm): {frequency_per_cm:.2f}{ i}", calculation, 1)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)


    import tuna_thermo as thermo

    log(f"\n Temperature used is {temperature:.2f} K, pressure used is {(pressure)} Pa.", calculation, 2)
    log(" Entropies multiplied by temperature to give units of energy.", calculation, 2)
    log(f" Using symmetry number derived from {point_group} point group for rotational entropy.", calculation, 2)

    #Calculates rotational constant for thermochemical calculations
    rotational_constant_per_cm, _ = postscf.calculate_rotational_constant(masses, coordinates)

    #Uses thermochemistry module to calculate various thermochemical properties
    U, translational_internal_energy, rotational_internal_energy, vibrational_internal_energy = thermo.calculate_internal_energy(energy, zpe, temperature, frequency_per_cm)
    H = thermo.calculate_enthalpy(U, temperature)
    S, translational_entropy, rotational_entropy, vibrational_entropy, electronic_entropy = thermo.calculate_entropy(temperature, frequency_per_cm, point_group, rotational_constant_per_cm * 100, masses, pressure)
    G = H - temperature * S

    #Makes sure decimal points are aligned in output
    space_1 = " " if U >= 0 else ""
    space_2 = " " if H >= 0 else ""
    space_3 = " " if energy >= G else ""

    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log("                                  Thermochemistry", calculation, 2)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log(f"  Electronic energy:    {energy:.10f}       Electronic entropy:      {temperature*electronic_entropy:.10f}", calculation, 2)
    log(f"\n  Zero-point energy:     {zpe:.10f}", calculation, 2)
    log(f"  Translational energy:  {translational_internal_energy:.10f}       Translational entropy:   {temperature*translational_entropy:.10f}", calculation, 2)
    log(f"  Rotational energy:     {rotational_internal_energy:.10f}       Rotational entropy:      {temperature*rotational_entropy:.10f}", calculation, 2)
    log(f"  Vibrational energy:    {vibrational_internal_energy:.10f}       Vibrational entropy:     {temperature*vibrational_entropy:.10f}  ", calculation, 2)
    log(f"\n  Internal energy:    {space_1}  {U:.10f}", calculation, 2)
    log(f"  Enthalpy:         {space_2}    {H:.10f}       Entropy:                 {temperature*S:.10f}", calculation, 2)
    log(f"\n  Gibbs free energy:    {G:.10f}       Non-electronic energy: {space_3} {energy - G:.10f}", calculation, 2)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
