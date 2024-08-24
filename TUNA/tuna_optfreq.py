import tuna_util as util
import numpy as np
import tuna_energy as energ
import sys
import tuna_postscf as postscf
from termcolor import colored

def print_trajectory(molecule, energy, coordinates):

        atoms = molecule.atoms

        with open("tuna-trajectory.xyz", "a") as file:
                
            file.write(f"{len(atoms)}\n")
            file.write(f"Coordinates from TUNA calculation. E = {energy:.10f}\n")

            coordinates_output = util.bohr_to_angstrom(coordinates)

            for i in range(len(atoms)):

                file.write(f"  {atoms[i]}      {coordinates_output[i][0]:6f}      {coordinates_output[i][1]:6f}      {coordinates_output[i][2]:6f}\n")

        file.close()


def calculate_gradient(coordinates, calculation, atoms):

    prod = 0.0001

    forward_coords = coordinates + np.array([[0, 0, 0], [0, 0, prod]])
    backward_coords = coordinates - np.array([[0, 0, 0], [0, 0, prod]])

    scf_output_forward, _ = energ.calculate_energy(calculation, atoms, forward_coords, silent=True)
    scf_output_backward, _ = energ.calculate_energy(calculation, atoms, backward_coords, silent=True)

    forward_energy = scf_output_forward.energy
    backward_energy = scf_output_backward.energy

    gradient = (forward_energy - backward_energy) / (2 * prod)

    return gradient
    


def calculate_approximate_hessian(delta_x, delta_grad): 

    hessian = delta_grad / delta_x

    return hessian



def calculate_hessian(energy, coordinates, calculation, atoms):
    
    prod = 0.0001

    far_forward_coords = coordinates + np.array([[0,0,0],[0,0, 2 * prod]])  
    forward_coords = coordinates + np.array([[0,0,0],[0,0, prod]])  
    backward_coords = coordinates - np.array([[0,0,0],[0,0, prod]])   
    far_backward_coords = coordinates - np.array([[0,0,0],[0,0, 2 * prod]])    

    scf_output_far_forward, _  = energ.calculate_energy(calculation, atoms, far_forward_coords, silent=True)   
    scf_output_forward, _  = energ.calculate_energy(calculation, atoms, forward_coords, silent=True)
    scf_output_backward, _  = energ.calculate_energy(calculation, atoms, backward_coords, silent=True)
    scf_output_far_backward, _  = energ.calculate_energy(calculation, atoms, far_backward_coords, silent=True)

    far_forward_energy = scf_output_far_forward.energy
    forward_energy = scf_output_forward.energy
    backward_energy = scf_output_backward.energy
    far_backward_energy = scf_output_far_backward.energy

    hessian = (-far_forward_energy + 16 * forward_energy - 30 * energy + 16 * backward_energy - far_backward_energy) / (12 * prod ** 2)

    return hessian



def optimise_geometry(calculation, atoms, starting_coordinates):
    
    maximum_step = util.angstrom_to_bohr(0.2)
    coordinates = starting_coordinates
    default_hessian = 1/4
    geom_conv_criteria = calculation.geom_conv
    max_geom_iter = calculation.geom_max_iter


    print("\nInitialising geometry optimisation...\n")

    if calculation.trajectory: 
        
        print("Printing trajectory data to \"tuna-trajectory.xyz\"\n")
        
        with open('tuna-trajectory.xyz', 'w'): pass

    if not calculation.calchess: print(f"Using approximate Hessian in convex region, Hessian of {default_hessian:.3f} outside.\n")
    else: print(f"Using exact Hessian in convex region, Hessian of {default_hessian:.3f} outside.\n")

    print(f"Gradient convergence: {geom_conv_criteria.get("gradient"):.7f}")
    print(f"Step convergence: {geom_conv_criteria.get("step"):.7f}")
    print(f"Maximum iterations: {max_geom_iter}")
    print(f"Maximum step: {util.bohr_to_angstrom(maximum_step):.5f}")

    P_guess = None; E_guess = 0

    for iteration in range(1, max_geom_iter + 1):

        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Beginning energy and gradient calculation on geometry iteration number {iteration}...")
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        
        if not calculation.moread: 

            P_guess = None
            E_guess = 0

        if calculation.additional_print: scf_output, molecule = energ.calculate_energy(calculation, atoms, coordinates, P_guess, E_guess, terse=False)
        else: scf_output, molecule = energ.calculate_energy(calculation, atoms, coordinates, P_guess, E_guess, terse=True)


        energy = scf_output.energy
        P = scf_output.P

        P_guess = P
        E_guess = energy


        print("\n Calculating numerical gradient...  ", end=""); sys.stdout.flush()
        gradient = calculate_gradient(coordinates, calculation, atoms)

        print("[Done]")

        bond_length = molecule.bond_length

        if gradient > 0: space = "  "
        else: space = "  "

        hessian = default_hessian
     
        if iteration > 1:

            if calculation.calchess: h = calculate_hessian(energy, coordinates, calculation, atoms)
            else: h = calculate_approximate_hessian(bond_length - old_bond_length, gradient - old_gradient)


            #Checks if region is convex or concave, if in the correct region for opt to min/max, sets the hessian to the second derivative
            if calculation.optmax:
                if h < 0.01: hessian = -h
            else: 
                if h > 0.01: hessian = h


        inverse_hessian = 1 / hessian
           
        step = inverse_hessian * gradient
        

        if np.abs(gradient) < geom_conv_criteria.get("gradient"): converged_grad = True; conv_check_grad = "Yes"
        else: converged_grad = False; conv_check_grad = "No"

        if np.abs(step) < geom_conv_criteria.get("step"): converged_step = True; conv_check_step = "Yes"
        else: converged_step = False; conv_check_step = "No"
        
        print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("   Factor       Value      Conv. Criteria    Converged")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"  Gradient    {gradient:.8f}  {space} {geom_conv_criteria.get("gradient"):.8f}   {space}    {conv_check_grad} ")
        print(f"    Step      {step:.8f}  {space} {geom_conv_criteria.get("step"):.8f}   {space}    {conv_check_step} ")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


        if calculation.trajectory: print_trajectory(molecule, energy, coordinates)


        if converged_grad and converged_step: 

            print("\n==========================================")           
            print(f" Optimisation converged in {iteration} iterations!")
            print("==========================================")

            postscf.post_scf_output(molecule, calculation, scf_output.epsilons, scf_output.molecular_orbitals, P, scf_output.S, scf_output.ao_ranges, scf_output.D, scf_output.P_alpha, scf_output.P_beta)
            
            print(f"\n Optimisation converged in {iteration} iterations to bond length of {util.bohr_to_angstrom(bond_length):.4f} Angstroms!")
            print(f"\n Final single point energy: {energy:.10f}")

            return molecule, energy

        else:
            
            if step > maximum_step: 
                step = maximum_step
                util.warning("Calculated step is outside of trust radius, taking maximum step instead.")

            elif step < -maximum_step:
                step = -maximum_step
                util.warning("Calculated step is outside of trust radius, taking maximum step instead.")

            if calculation.optmax: direction = -1
            else: direction = 1

            coordinates = np.array([[0, 0, 0], [0, 0, coordinates[1][2] - direction * step]])

            if coordinates[1][2] <= 0: util.error("Optimisation generated negative bond length! Decrease trust radius!")

            old_bond_length = bond_length
            old_gradient = gradient
     

    util.warning(F"Geometry optimisation did not converge in {max_geom_iter} iterations! Increase the maximum or give up!")





def calculate_frequency(calculation, atoms=None, coordinates=None, optimised_molecule=None, optimised_energy=None):

    if calculation.calculation_type == "FREQ":
          
        scf_output, molecule = energ.calculate_energy(calculation, atoms, coordinates)
        energy = scf_output.energy
    
    else:

        molecule = optimised_molecule
        energy = optimised_energy


    point_group = molecule.point_group
    bond_length = molecule.bond_length
    atoms = molecule.atoms
    coordinates = molecule.coordinates
    masses = molecule.masses

    temp = calculation.temperature
    pres = calculation.pressure  


    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Beginning TUNA harmonic frequency calculation...")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    
    print(f"\n Calculating numerical Hessian at bond length of {util.bohr_to_angstrom(bond_length):.4f} Angstroms...  ", end=""); sys.stdout.flush()
    
    k = calculate_hessian(energy, coordinates, calculation, atoms)
    
    print("[Done]\n")
    
    
    reduced_mass = postscf.calculate_reduced_mass(masses)

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
        
        util.warning("Imaginary frequency calculated! Zero-point energy and vibrational thermochemistry set to zero!")


    frequency_per_cm = frequency_hartree * util.constants.per_cm_in_hartree
    


    print(" Using masses of most abundant isotopes...\n")

    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("       Harmonic Frequency")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"  Force constant: {k:.5f}")
    print(f"  Reduced mass: {reduced_mass:.2f}")
    print(f"\n  Frequency (per cm): {frequency_per_cm:.2f}{ i}")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    import tuna_thermo as thermo
    

    print(f"\n Temperature used is {temp:.2f} K, pressure used is {(pres)} Pa.")
    print(" Entropies multiplied by temperature to give units of energy.")
    print(f" Using symmetry number derived from {point_group} point group for rotational entropy.")

    rotational_constant_per_cm, rotational_constant_GHz = postscf.calculate_rotational_constant(masses, coordinates)

    U, translational_internal_energy, rotational_internal_energy, vibrational_internal_energy = thermo.calculate_internal_energy(energy, zpe, temp, frequency_per_cm)
    H = thermo.calculate_enthalpy(U, temp)

    S, translational_entropy, rotational_entropy, vibrational_entropy, electronic_entropy = thermo.calculate_entropy(temp, frequency_per_cm, point_group, rotational_constant_per_cm * 100, masses, pres)
    G = H - temp * S


    if U >= 0: space_1 = " "
    else: space_1 = ""
    if H >= 0: space_2 = " "
    else: space_2 = ""

    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("                                  Thermochemistry")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"  Electronic energy:    {energy:.10f}       Electronic entropy:      {temp*electronic_entropy:.10f}")
    print(f"\n  Zero-point energy:     {zpe:.10f}")
    print(f"  Translational energy:  {translational_internal_energy:.10f}       Translational entropy:   {temp*translational_entropy:.10f}")
    print(f"  Rotational energy:     {rotational_internal_energy:.10f}       Rotational entropy:      {temp*rotational_entropy:.10f}")
    print(f"  Vibrational energy:    {vibrational_internal_energy:.10f}       Vibrational entropy:     {temp*vibrational_entropy:.10f}  ")
    print(f"\n  Internal energy:    {space_1}  {U:.10f}")
    print(f"  Enthalpy:         {space_2}    {H:.10f}       Entropy:                 {temp*S:.10f}")
    print(f"\n  Gibbs free energy:    {G:.10f}      Non-electronic energy:   {energy - G:.10f}")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
