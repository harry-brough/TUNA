import __init__ as init
version_number = init.__version__
from termcolor import colored

#Prints big fish logo
print(colored("\n      _______ _    _ _   _                     ___           \n     |__   __| |  | | \\ | |   /\\            __/__/__  _      \n","white")+ colored("~~~~~~","light_grey")+colored("  | |  | |  | |  \\| |  /  \\","white")+colored(" ~~~~~~~~","light_grey")+colored(" / .      \\/ ) ","white")+colored("~~~~\n ~~~~~~","light_grey")+colored(" | |  | |  | | . ` | / /\\ \\","white")+colored(" ~~~~~~","light_grey")+colored(" (     ))    (","white")+colored(" ~~~~~\n ~~~~~~","light_grey")+colored(" | |  | |__| | |\\  |/ ____ \\ ","white")+colored("~~~~~~","light_grey")+colored(" \\___  ___/\\_) ","white")+colored("~~~~","light_grey")+colored("\n        |_|   \\____/|_| \\_/_/    \\_\\          \\\\_\\           ", "white"))
print("\n")

print(f"Welcome to version {version_number} of TUNA (Theoretical Unification of Nuclear Arrangements)!\n")
print("Importing required libraries...    ",end="")

import sys; sys.stdout.flush()
import numpy as np
import time
from tuna_util import *
import tuna_energy as energ
import tuna_optfreq as optfreq
import tuna_md as md


print("[Done]\n")

start_time = time.perf_counter()


def parse_input():

    """

    Reads the input line specified by the user, and returns the type of calculation requested, method, basis set, atoms in the molecule and their coordinates, and
    other parameters. Contains various error handling methods so even I can't break it.

    Returns calculation type (string), method (string), basis set (string), atoms (list), coordinates (array), and parameters (list).

    """

    #Allowed options for the input line
    atom_options = ["XH", "XHE", "H", "HE"]
    calculation_options = ["SPE", "OPT", "SCAN", "FREQ", "OPTFREQ", "MD", "ANHARM"]
    method_options = ["HF", "RHF", "UHF", "MP2", "SCS-MP2", "UMP2", "MP3", "UMP3", "SCS-MP3"]
    basis_options = ["STO-3G", "STO-6G", "3-21G", "4-31G", "6-31G", "6-31+G", "6-31++G", "6-311G", "6-311+G", "6-311++G", "HTO-CBS"]

    #Puts input line into standardised format, capital letters and separated by spaces for each argument
    input_line = " ".join(sys.argv[1:]).upper().strip()

    try: 
        
        #Separated input line into sections separated by a colon, extracts relevant information from those sections
        sections = input_line.split(":")

        calculation_type = sections[0].strip()
        geometry_section = sections[1].strip()
        method, basis = sections[2].strip().split()

        if len(sections) == 4: params = sections[3].strip().split()  
        else: params = []

        for param in params: param = param.strip()   

    except: error("Input line formatted incorrectly! Read the manual for help.")

    #Creates a list of atoms, either one or two long
    atoms = [atom.strip() for atom in geometry_section.split(" ")[0:2] if atom.strip()]
    
    try:
        
        #Extracts bond length from geometry section if it exists, sets coordinates to be [0, bond length]
        coordinates_1d = [0] + [float(bond_length.strip()) for bond_length in geometry_section.split(" ")[2:] if bond_length.strip()]
    
    except ValueError: error("Could not parse bond length!")
    
    #Checks if requested calculation, method, basis, etc. are in the allowed options, then gives relevant error message if not 
    if calculation_type not in calculation_options: error(f"Calculation type \"{calculation_type}\" is not supported.")
    if method not in method_options: error(f"Calculation method \"{method}\" is not supported.")
    if basis not in basis_options: error(f"Basis set \"{basis}\" is not supported.")
    if not all(atom in atom_options for atom in atoms): error("One or more atom types not recognised! Available atoms are H, He and ghost atoms XH and XHe")
    if len(atoms) != len(coordinates_1d): error("Two atoms requested without a bond length!")

    #Rejects requests for tiny bond lengths, such as two atoms on top of each other
    if len(coordinates_1d) == 2 and coordinates_1d[1] < 0.05: error(f"Bond length ({coordinates_1d[1]} angstroms) too small! Minimum bond length is 0.05 angstroms.")

    #Converts 1D coordinate array in angstroms to 3D array ion bohr
    coordinates = one_dimension_to_three(angstrom_to_bohr(np.array(coordinates_1d)))


    return calculation_type, method, basis, atoms, coordinates, params




def run_calculation(calculation_type, calculation, atoms, coordinates):

    """
    
    Sets off the requested calculation, using the information provided by the user.

    Requires calculation type (string), calculation (Calculation), atoms (list), and coordinates (array).

    """


    if calculation_type == "SPE": 
        
        #Simply calculates energy
        energ.calculate_energy(calculation, atoms, coordinates)

    elif calculation_type == "SCAN":

        #Turns off reading the density matrix from previous step only if NOMOREAD parameter has been used
        if calculation.nomoread: calculation.moread = False

        #Sets off coordinate scan only if the number of step and step size have been given, otherwise throws a relevant error
        if calculation.scanstep:
            if calculation.scannumber: 
                
                energ.scan_coordinate(calculation, atoms, coordinates)
                
            else: error(f"Coordinate scan requested but no number of steps given by keyword \"SCANNUMBER\"!")
        else:  error(f"Coordinate scan requested but no step size given by keyword \"SCANSTEP\"!")
        

    elif calculation_type == "OPT":
        
        #Turns off reading the density matrix from previous step only if NOMOREAD parameter has been used
        if calculation.nomoread: calculation.moread = False

        #Makes sure there are multiple atoms, then optimises the geometry
        if len(atoms) == 1 or any("X" in atom for atom in atoms): error("Geometry optimisation requested for single atom!")
        
        optfreq.optimise_geometry(calculation, atoms, coordinates)


    elif calculation_type == "FREQ":

        #Makes sure there are multiple atoms, then calculates the frequency
        if len(atoms) == 1 or any("X" in atom for atom in atoms): error("Harmonic frequencies requested for single atom!")

        optfreq.calculate_frequency(calculation, atoms=atoms, coordinates=coordinates)


    elif calculation_type == "OPTFREQ":

        #Turns off reading the density matrix from previous step only if NOMOREAD parameter has been used
        if calculation.nomoread: calculation.moread = False

        #Makes sure there are multiple atoms, then optimises the geometry and subsequently uses the optimised structure to find the harmonic frequency
        if len(atoms) == 1 or any("X" in atom for atom in atoms): error("Geometry optimisation requested for single atom!")

        optimised_molecule, optimised_energy = optfreq.optimise_geometry(calculation, atoms, coordinates)
        optfreq.calculate_frequency(calculation, optimised_molecule=optimised_molecule, optimised_energy=optimised_energy)
        

    elif calculation_type == "MD":

        #Turns off reading the density matrix from previous step only if NOMOREAD parameter has been used
        if calculation.nomoread: calculation.moread = False

        #Turns on printing the trajectory only if NOTRAJ parameter has not been used
        if not calculation.notrajectory: calculation.trajectory = True

        #Makes sure there are multiple atoms, then starts the molecular dynamics simulation
        if len(atoms) == 1 or any("X" in atom for atom in atoms): error("Molecular dynamics calculation requested for single atom!")

        md.run_md(calculation, atoms, coordinates)
        



def main(): 

    """
    
    Sets off TUNA, first parsing the input line, then printing the key information, building the calculation object with all the relevant parameters,
    running the desired calculation and finally finishing the calculation.

    """

    #Reads input line, makes sure it's okay and extracts the desired parameters
    calculation_type, method, basis, atoms, coordinates, params = parse_input()

    print(f"{calculation_types.get(calculation_type)} calculation in \"{basis}\" basis set via {method_types.get(method)} requested.")

    #Builds calculation object which holds onto all the fundamental and derived parameters, passed through most functions in TUNA
    calculation = Calculation(calculation_type, method, start_time, params, basis)

    #If a decontracted basis has been requested, this is printed to the console
    if calculation.decontract: print("Setting up calculation using fully decontracted basis set.")
    else: print("Setting up calculation using partially contracted basis set.")

    print(f"\nDistances in angstroms and times in femtoseconds. Everything else in atomic units.")

    #Sets off the desired calculation with the requested parameters
    run_calculation(calculation_type, calculation, atoms, coordinates)

    #Finishes the calculation, printing the time taken
    finish_calculation(calculation)



if __name__ == "__main__": main()



"""
TODO
Simplify the MPN stuff by making it all spin orbital only
Just transform the non antisymmetrised integrals, then do SCS, and antisymmetrise after
Harmonic frequency intensity, dipole derivatives
Add anharmonic frequencies
Harpy 1rdm for (oo)UMP2
Copy HarPy integrals
Add CCSD
"""
