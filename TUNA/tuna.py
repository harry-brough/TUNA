import __init__ as init
version_number = init.__version__
from termcolor import colored

print(colored("\n      _______ _    _ _   _                     ___           \n     |__   __| |  | | \\ | |   /\\            __/__/__  _      \n","white")+ colored("~~~~~~","light_grey")+colored("  | |  | |  | |  \\| |  /  \\","white")+colored(" ~~~~~~~~","light_grey")+colored(" / .      \\/ ) ","white")+colored("~~~~\n ~~~~~~","light_grey")+colored(" | |  | |  | | . ` | / /\\ \\","white")+colored(" ~~~~~~","light_grey")+colored(" (     ))    (","white")+colored(" ~~~~~\n ~~~~~~","light_grey")+colored(" | |  | |__| | |\\  |/ ____ \\ ","white")+colored("~~~~~~","light_grey")+colored(" \\___  ___/\\_) ","white")+colored("~~~~","light_grey")+colored("\n        |_|   \\____/|_| \\_/_/    \\_\\          \\\\_\\           ", "white"))
print("\n")

print(f"Welcome to version {version_number} of TUNA (Theoretical Unification of Nuclear Arrangements)!\n")
print("Importing required libraries...    ",end="")

import sys; sys.stdout.flush()
import numpy as np
import time
import tuna_util as util
import tuna_energy as energ
import tuna_optfreq as optfreq
import tuna_md as md


print("[Done]\n")

start_time = time.perf_counter()


def parse_input():

    atom_options = ["XH", "XHE", "H", "HE"]
    calculation_options = ["SPE", "OPT", "SCAN", "FREQ", "OPTFREQ", "MD", "ANHARM"]
    method_options = ["HF", "RHF", "UHF", "MP2", "SCS-MP2", "UMP2", "MP3", "UMP3", "SCS-MP3"]
    basis_options = ["STO-3G", "STO-6G", "3-21G", "4-31G", "6-31G", "6-31+G", "6-31++G", "6-311G", "6-311+G", "6-311++G", "HTO-CBS"]

    input_line = " ".join(sys.argv[1:]).upper().strip()

    try: 
        
        sections = input_line.split(":")

        calculation_type = sections[0].strip()
        geometry_section = sections[1].strip()
        method, basis = sections[2].strip().split()

        if len(sections) == 4: params = sections[3].strip().split()  
        else: params = []

        for param in params: param = param.strip()   

    except: util.error("Input line formatted incorrectly! Read the manual for help.")


    atoms = [atom.strip() for atom in geometry_section.split(" ")[0:2] if atom.strip()]
    
    try:
    
        coordinates_1d = [0] + [float(bond_length.strip()) for bond_length in geometry_section.split(" ")[2:] if bond_length.strip()]
    
    except ValueError: util.error("Could not parse bond length!")
    
    if calculation_type not in calculation_options: util.error(f"Calculation type \"{calculation_type}\" is not supported.")
    if method not in method_options: util.error(f"Calculation method \"{method}\" is not supported.")
    if basis not in basis_options: util.error(f"Basis set \"{basis}\" is not supported.")
    if not all(atom in atom_options for atom in atoms): util.error("One or more atom types not recognised! Available atoms are H, He and ghost atoms XH and XHe")
    if len(atoms) != len(coordinates_1d): util.error("Two atoms requested without a bond length!")

    if len(coordinates_1d) == 2 and coordinates_1d[1] < 0.05: util.error(f"Bond length ({coordinates_1d[1]} angstroms) too small! Minimum bond length is 0.05 angstroms.")

    coordinates = util.one_dimension_to_three(util.angstrom_to_bohr(np.array(coordinates_1d)))


    return calculation_type, method, basis, atoms, coordinates, params


def run_calculation(calculation_type, calculation, atoms, coordinates):

    if calculation_type == "SPE": 
        
        energ.calculate_energy(calculation, atoms, coordinates)

    elif calculation_type == "SCAN":

        if calculation.nomoread: calculation.moread = False

        if calculation.scanstep:
            if calculation.scannumber: 
                
                energ.scan_coordinate(calculation, atoms, coordinates)
                
            else: util.error(f"Coordinate scan requested but no number of steps given by keyword \"SCANNUMBER\"!")
        else:  util.error(f"Coordinate scan requested but no step size given by keyword \"SCANSTEP\"!")
        

    elif calculation_type == "OPT":
        
        if calculation.nomoread: calculation.moread = False

        if len(atoms) == 1 or "XH" in atoms or "XHE" in atoms: util.error("Geometry optimisation requested for single atom!")
        
        optfreq.optimise_geometry(calculation, atoms, coordinates)


    elif calculation_type == "FREQ":

        if len(atoms) == 1 or "XH" in atoms or "XHE" in atoms: util.error("Harmonic frequencies requested for single atom!")

        optfreq.calculate_frequency(calculation, atoms=atoms, coordinates=coordinates)


    elif calculation_type == "OPTFREQ":

        if calculation.nomoread: calculation.moread = False

        if len(atoms) == 1 or "XH" in atoms or "XHE" in atoms: util.error("Geometry optimisation requested for single atom!")

        optimised_molecule, optimised_energy = optfreq.optimise_geometry(calculation, atoms, coordinates)
        optfreq.calculate_frequency(calculation, optimised_molecule=optimised_molecule, optimised_energy=optimised_energy)
        

    elif calculation_type == "MD":

        if calculation.nomoread: calculation.moread = False
        if not calculation.notrajectory: calculation.trajectory = True

        if len(atoms) == 1 or "XH" in atoms or "XHE" in atoms: util.error("Molecular dynamics calculation requested for single atom!")

        md.run_md(calculation, atoms, coordinates)
        

def main(): 

    calculation_type, method, basis, atoms, coordinates, params = parse_input()


    print(f"{util.calculation_types.get(calculation_type)} calculation in \"{basis}\" basis set via {util.method_types.get(method)} requested.")

    calculation = util.Calculation(calculation_type, method, start_time, params, basis)

    if calculation.decontract: print("Setting up calculation using fully decontracted basis set.")
    else: print("Setting up calculation using partially contracted basis set.")

    print(f"\nDistances in angstroms and times in femtoseconds. Everything else in atomic units.")

    run_calculation(calculation_type, calculation, atoms, coordinates)

    util.finish_calculation(calculation)



if __name__ == "__main__": main()



"""
TODO

Harmonic frequency intensity, dipole derivatives
Add comments and docstrings
Redo logging
Add MOREAD to MD
Add anharmonic frequencies
Change rotation in MD, use unitary force vector instead by coords[1] - coords[0] / norm, then multiply by magnitude
Add CCSD
Copy QChem Mulliken formulas
Confirm energy components always work
Capitalise Xh Xhe
Look into DIIS for UHF with rotate, compare to ORCA
Harpy 1rdm for (oo)UMP2
Copy HarPy integrals
Just transform the non antisymmetrised integrals, then do SCS, and antisymmetrise after
Investigate whether the rotation angle should be 45 or 90 or what
"""