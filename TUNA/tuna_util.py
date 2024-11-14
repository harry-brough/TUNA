import numpy as np
import time, sys
import tuna_basis as basis_sets
from termcolor import colored

calculation_types = {"SPE": "Single point energy", "OPT": "Geometry optimisation", "FREQ": "Harmonic frequency", "OPTFREQ": "Optimisation and harmonic frequency", "SCAN": "Coordinate scan", "MD": "Ab initio molecular dynamics", "ANHARM": "Anharmonic frequency"}
method_types = {"HF": "Hartree-Fock theory", "RHF": "restricted Hartree-Fock theory", "UHF": "unrestricted Hartree-Fock theory", "MP2": "MP2 theory", "UMP2": "unrestricted MP2 theory", "SCS-MP2": "spin-component-scaled MP2 theory", "MP3": "MP3 theory", "UMP3": "unrestricted MP3 theory", "SCS-MP3": "spin-component-scaled MP3 theory"}


class Constants:

    """
    
    Defines all the constants used in TUNA, from the fundamental constants used, including atomic masses, and emergent 
    constants and conversion factors for units used throughout the program. The fundamental values are taken from the
    CODATA 2022 recommended values.

    """

    def __init__(self):

        #Fundamental constants to define Hartree land
        self.planck_constant_in_joules_seconds = 6.62607015e-34
        self.elementary_charge_in_coulombs = 1.602176634e-19
        self.electron_mass_in_kilograms = 9.1093837139e-31
        self.permittivity_in_farad_per_metre = 1.11265005620e-10

        #More fundamental constants
        self.c_in_metres_per_second = 299792458
        self.k_in_joules_per_kelvin = 1.380649e-23
        self.atomic_mass_unit_in_kg = 1.660539068911e-27
        self.reduced_planck_constant_in_joules_seconds = self.planck_constant_in_joules_seconds / (2 * np.pi)

        #Emergent unit conversions
        self.bohr_in_metres = self.permittivity_in_farad_per_metre * self.reduced_planck_constant_in_joules_seconds ** 2 / (self.electron_mass_in_kilograms * self.elementary_charge_in_coulombs ** 2)
        self.hartree_in_joules = self.reduced_planck_constant_in_joules_seconds ** 2 / (self.electron_mass_in_kilograms * self.bohr_in_metres ** 2)
        self.atomic_time_in_seconds = self.reduced_planck_constant_in_joules_seconds /  self.hartree_in_joules
        self.atomic_time_in_femtoseconds = self.atomic_time_in_seconds * 10 ** 15
        self.bohr_radius_in_angstrom = self.bohr_in_metres * 10 ** 10
        self.pascal_in_atomic_units = self.hartree_in_joules / self.bohr_in_metres ** 3
        self.per_cm_in_hartree = self.hartree_in_joules / (self.c_in_metres_per_second * self.planck_constant_in_joules_seconds * 10 ** 2)
        self.per_cm_in_GHz = self.hartree_in_joules / (self.planck_constant_in_joules_seconds * self.per_cm_in_hartree * 10 ** 9)
        self.atomic_mass_unit_in_electron_mass = self.atomic_mass_unit_in_kg / self.electron_mass_in_kilograms

        #Emergent constants
        self.c = self.c_in_metres_per_second * self.atomic_time_in_seconds / self.bohr_in_metres
        self.k = self.k_in_joules_per_kelvin / self.hartree_in_joules
        self.h = self.planck_constant_in_joules_seconds / (self.hartree_in_joules * self.atomic_time_in_seconds)

        #Atomic masses
        self.atom_masses_in_amu = {"H": 1.00782503223, "HE": 4.00260325413}
        self.atom_masses = {"H": self.atom_masses_in_amu.get("H") * self.atomic_mass_unit_in_electron_mass, "HE": self.atom_masses_in_amu.get("HE") * self.atomic_mass_unit_in_electron_mass}




class Calculation:

    """
    
    Holds onto dictionaries that define all the convergence criteria, and calculates various parameters from the user-defined parameters
    specified when the TUNA calculation begins. Various default values for parameters are also specified here. This object is created only
    once, at the start of a TUNA calculation.
    
    """

    loose = {"delta_E": 0.000001, "maxDP": 0.00001, "rmsDP": 0.000001, "orbitalGrad": 0.0001, "word": "loose"}
    normal = {"delta_E": 0.0000001, "maxDP": 0.000001, "rmsDP": 0.0000001, "orbitalGrad": 0.00001, "word": "medium"}
    tight = {"delta_E": 0.000000001, "maxDP": 0.00000001, "rmsDP": 0.000000001, "orbitalGrad": 0.0000001, "word": "tight"}
    extreme = {"delta_E": 0.00000000001, "maxDP": 0.0000000001, "rmsDP": 0.00000000001, "orbitalGrad": 0.000000001, "word": "extreme"}   

    looseopt = {"gradient": 0.001, "step": 0.01, "word": "loose"}
    normalopt = {"gradient": 0.0001, "step": 0.0001, "word": "medium"}
    tightopt = {"gradient": 0.000001, "step": 0.00001, "word": "tight"}
    extremeopt = {"gradient": 0.00000001, "step": 0.0000001, "word": "extreme"}



    def __init__(self, calculation_type, method, start_time, params, basis):

        #Key calculation parameters
        self.calculation_type = calculation_type
        self.method = method
        self.start_time = start_time
        self.basis = basis
        self.mp2_type = None
        
        #Secondary important factors to begin a calculation
        self.norotate_guess = False
        self.rotate_guess = False
        self.decontract = False
        self.theta = np.pi / 4
        
        #Sets "tight" convergence criteria if calculation type is OPT, FREQ, OPTFREQ or MD, otherwise uses normal criteria
        if self.calculation_type == "OPT" or self.calculation_type == "FREQ" or self.calculation_type == "OPTFREQ" or self.calculation_type == "MD": self.scf_conv = self.tight
        else: self.scf_conv = self.normal
        
        #Convergence default parameters
        self.level_shift = False
        self.level_shift_parameter = 0.2

        #Default parameters for geometry optimisation
        self.geom_conv = self.tightopt

        #Temperature set to 0 K for MD, 298.15 K for FREQ
        if self.calculation_type == "MD": self.temperature = 0
        else: self.temperature = 298.15


        #Process the user-defined parameters
        self.process_params(params)


    def process_params(self, params):
        
        """
        
        Requires parameters (list).

        Processes parameters, overriding default values if these are provided.

        """

        # Simple parameter processing

        self.diis = True if "DIIS" in params else True
        self.diis = False if "NODIIS" in params else True
        self.damping = True if "DAMP" in params else True
        self.damping = False if "NODAMP" in params else True
        self.densplot = True if "DENSPLOT" in params else False
        self.scanplot = True if "SCANPLOT" in params else False
        self.d2 = True if "D2" in params else False
        self.calchess = True if "CALCHESS" in params else False
        self.additional_print = True if "P" in params else False
        self.terse = True if "T" in params else False
        self.moread_requested = True if "MOREAD" in params else True
        self.nomoread = True if "NOMOREAD" in params else False 
        self.optmax = True if "OPTMAX" in params else False
        self.trajectory = True if "TRAJ" in params else False
        self.notrajectory = True if "NOTRAJ" in params else False
        self.slowconv = True if "SLOWCONV" in params else False
        self.veryslowconv = True if "VERYSLOWCONV" in params else False
        self.no_levelshift = True if "NOLEVELSHIFT" in params else False

        if "LEVELSHIFT" in params: 

            self.level_shift = True

            try:

                params.index("LEVELSHIFT")
                self.level_shift_parameter = float(params[params.index("LEVELSHIFT") + 1])
        
            except:
                pass
        

        
        if "ROTATE" in params: 

            self.rotate_guess = True

            try:

                params.index("ROTATE")
                self.theta = float(params[params.index("ROTATE") + 1]) * np.pi / 180
        
            except:
                pass

        elif "NOROTATE" in params: self.norotate_guess = True
        self.moread = True     
        
        # Convergence criteria for SCF
        if "LOOSE" in params or "LOOSESCF" in params: self.scf_conv = self.loose  
        elif "NORMAL" in params or "NORMALSCF" in params: self.scf_conv = self.normal  
        elif "TIGHT" in params or "TIGHTSCF" in params: self.scf_conv = self.tight   
        elif "EXTREME" in params or "EXTREMESCF" in params: self.scf_conv = self.extreme 

        # Convergence criteria for geometry optimisation
        if "LOOSEOPT" in params: self.geom_conv = self.looseopt  
        elif "NORMALOPT" in params: self.geom_conv = self.normalopt  
        elif "TIGHTOPT" in params: self.geom_conv = self.tightopt 
        elif "EXTREMEOPT" in params: self.geom_conv = self.extremeopt    

        # Automates error messages for parameters with required variables
        def get_param_value(param_name, value_type):

            if param_name in params:

                try: return value_type(params[params.index(param_name) + 1])
                except IndexError: error(f"Parameter \"{param_name}\" requested but no value specified!")
                except ValueError: error(f"Parameter \"{param_name}\" must be of type {value_type.__name__}!")
            
            return 
        
        # Key parameters
        self.charge = get_param_value("CHARGE", int) if get_param_value("CHARGE", int) is not None else 0
        self.charge = get_param_value("CH", int) if get_param_value("CH", int) is not None else 0 
        self.multiplicity = get_param_value("MULTIPLICITY", int) if get_param_value("MULTIPLICITY", int) is not None else 1
        self.multiplicity = get_param_value("ML", int) if get_param_value("ML", int) is not None else 1
        self.default_multiplicity = False if "ML" in params or "MULTIPLICITY" in params else True

        # Optimisation, coordinate scan and MD parameters
        self.max_iter = get_param_value("MAXITER", int) or 50
        self.max_step = get_param_value("MAXSTEP", float) or 0.2
        self.default_hessian = get_param_value("DEFAULTHESS", float) or 1 / 4
        self.geom_max_iter = get_param_value("GEOMMAXITER", int) or 30
        self.geom_max_iter = get_param_value("MAXGEOMITER", int) or 30
        self.scanstep = get_param_value("SCANSTEP", float) or None
        self.scannumber = get_param_value("SCANNUMBER", int) or None
        self.md_number_of_steps = get_param_value("MDNUMBER", int) or 50
        self.timestep = get_param_value("TIMESTEP", float) or 0.1

        # Thermochemical parameters
        self.temperature = get_param_value("TEMP", float) or self.temperature
        self.temperature = get_param_value("TEMPERATURE", float) or self.temperature
        self.pressure = get_param_value("PRES", float) or 101325
        self.pressure = get_param_value("PRESSURE", float) or 101325




class Molecule:

    """
    
    Calculates and holds onto various useful molecular information and quantities. This object can be created multiple times per
    calculation, such as every new geometry in a geometry optimisation or MD simulation.
    
    """


    def __init__(self, atoms, coordinates, calculation):

        # Charges and masses of atoms
        atom_charges = {"XH": 0, "XHE": 0, "H": 1, "HE": 2}
        atom_masses = Constants().atom_masses

        self.atoms = atoms
        self.masses = np.array([atom_masses[atom] for atom in self.atoms if "X" not in atom])
        self.charges = [atom_charges[atom] for atom in self.atoms]

        # Key molecular parameters
        self.coordinates = coordinates
        self.charge = calculation.charge
        self.multiplicity = calculation.multiplicity
        self.basis = calculation.basis

        self.n_electrons = np.sum(self.charges) - self.charge

        self.point_group = self.determine_point_group()
        self.molecular_structure = self.determine_molecular_structure()

        # Integral and related data
        self.mol = [basis_sets.generate_atomic_orbitals(atom, self.basis, coord) for atom, coord in zip(self.atoms, self.coordinates)]    
        self.ao_ranges = [len(basis_sets.generate_atomic_orbitals(atom, self.basis, coord)) for atom, coord in zip(self.atoms, self.coordinates)]
        self.atomic_orbitals = [orbital for atom_orbitals in self.mol for orbital in atom_orbitals] 
        self.pgs = [pg for atomic_orbital in self.atomic_orbitals for pg in atomic_orbital]

        # Decontracts orbitals if DECONTRACT keyword is requested
        if calculation.decontract: self.atomic_orbitals = [[pg] for pg in self.pgs]

        # If a molecule is supplied, calculate the bond length and centre of mass
        if len(self.atoms) == 2: 
            
            self.bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

            if not any("X" in atom for atom in self.atoms):

                self.centre_of_mass = calculate_centre_of_mass(self.masses, self.coordinates)

            else: self.centre_of_mass = 0

        else: 

            self.bond_length = "N/A"
            self.centre_of_mass = 0

        # If multiplicity not specified bu molecule has an odd number of electrons, set it to a doublet
        if calculation.default_multiplicity and self.n_electrons % 2 != 0: self.multiplicity = 2

        # Set the reference determinant to be used
        if calculation.method == "UHF": calculation.reference = "UHF"
        elif calculation.method == "RHF": calculation.reference = "RHF"
        elif calculation.method == "HF" and self.multiplicity == 1: calculation.reference = "RHF"
        elif calculation.method == "HF" and self.multiplicity != 1: calculation.reference = "UHF"

        elif calculation.method == "UMP2": 
            calculation.reference = "UHF"
            calculation.mp2_basis = "SO"
        elif calculation.method in ["MP2", "SCS-MP2", "SCS-MP3"] and self.multiplicity == 1: 
            calculation.reference = "RHF"
            calculation.mp2_basis = "MO"
        elif calculation.method == "MP2" and self.multiplicity != 1: 
            calculation.reference = "UHF"
            calculation.mp2_basis = "SO"
        elif calculation.method == "SCS-MP2" and self.multiplicity != 1:

            if self.n_electrons != 1: error("Using unrestricted references with SCS-MP2 is not supported yet!")
            else: calculation.reference = "RHF"; calculation.mp2_basis = "MO"

        elif calculation.method == "SCS-MP3" and self.multiplicity != 1:

            if self.n_electrons != 1: error("Using unrestricted references with SCS-MP3 is not supported yet!")
            else: calculation.reference = "RHF"; calculation.mp2_basis = "MO"

        elif calculation.method == "UMP3": calculation.reference = "UHF"
        elif calculation.method == "RMP3": calculation.reference = "RHF"
        elif calculation.method == "MP3" and self.multiplicity == 1: calculation.reference = "RHF"
        elif calculation.method == "MP3" and self.multiplicity != 1: calculation.reference = "UHF"

        # Sets the MP2 type to spin orbital if MP3 is requested
        if calculation.method in ["MP3", "RMP3", "UMP3"]: calculation.mp2_basis = "SO" 
    
        # Sets information about alpha and beta electrons for UHF
        self.n_unpaired_electrons = self.multiplicity - 1
        self.n_alpha = int((self.n_electrons + self.n_unpaired_electrons) / 2)
        self.n_beta = int(self.n_electrons - self.n_alpha)
        self.n_doubly_occ = min(self.n_alpha, self.n_beta)
        self.n_occ = self.n_alpha + self.n_beta

        # Sets off errors for invalid molecular configurations
        if self.n_electrons % 2 == 0 and self.multiplicity % 2 == 0: error("Impossible charge and multiplicity combination (both even)!")
        if self.n_electrons % 2 != 0 and self.multiplicity % 2 != 0: error("Impossible charge and multiplicity combination (both odd)!")
        if self.n_electrons - self.multiplicity < -1: error("Multiplicity too high for number of electrons!")
        if self.multiplicity < 1: error("Multiplicity must be at least 1!")

        # Sets off errors for invalid use of restricted Hartree-Fock
        if calculation.reference == "RHF":

            if self.n_electrons % 2 != 0: error("Restricted Hartree-Fock is not compatible with an odd number of electrons!")
            if self.multiplicity != 1: error("Restricted Hartree-Fock is not compatible non-singlet states!")

        # Sets 2 electrons per orbital for RHF, otherwise 1 for UHF
        calculation.n_electrons_per_orbital = 2 if calculation.reference == "RHF" else 1

        calculation.moread = False if calculation.reference == "UHF" and not calculation.moread_requested else True  


    def determine_point_group(self):

        """
        
        Determines and returns the point group based on the symmetry of the molecule (string).
        
        """

        # Same atom -> Dinfh, two different atoms -> Cinfv, single atom -> K
        if len(self.atoms) == 2 and "X" not in self.atoms[0] and "X" not in self.atoms[1]:
       
            return "Dinfh" if self.atoms[0] == self.atoms[1] else "Cinfv"

        if "X" in self.atoms[0] and "X" in self.atoms[1]: return "None"

        return "K"




    def determine_molecular_structure(self):

        """
        
        Builds and returns molecular structure (string), formatted nicely.
        
        """

        if len(self.atoms) == 2:
            
            if "X" not in self.atoms[0] and "X" not in self.atoms[1]: return f"{self.atoms[0].lower().capitalize()} --- {self.atoms[1].lower().capitalize()}"
            elif "X" in self.atoms[0] and "X" in self.atoms[1]: return "None" 

            elif "X" in self.atoms[0]: return f"{self.atoms[1].lower().capitalize()}"
            elif "X" in self.atoms[1]: return f"{self.atoms[0].lower().capitalize()}"

        return self.atoms[0].lower().capitalize()



class Output:

    """
    
    Builds an object to hold onto all useful parameters produced by a converged SCF calculation.
    
    """

    def __init__(self, energy, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta, epsilons, epsilons_alpha, epsilons_beta):
        
        #Key quantities
        self.energy = energy
        self.S = S

        #Density matrices
        self.P = P
        self.P_alpha = P_alpha
        self.P_beta = P_beta

        #Molecular orbitals
        self.molecular_orbitals = molecular_orbitals
        self.molecular_orbitals_alpha = molecular_orbitals_alpha
        self.molecular_orbitals_beta = molecular_orbitals_beta

        #Eigenvalues
        self.epsilons = epsilons
        self.epsilons_alpha = epsilons_alpha
        self.epsilons_beta = epsilons_beta
        self.epsilons_combined = np.append(self.epsilons_alpha, self.epsilons_beta)



#Initialises a new set of constants for use throughout the whole program - this is probably bad practice
constants = Constants()



def rotate_coordinates_to_z_axis(difference_vector):

    """
    
    Requires difference vector (array).

    Calculates the axis of rotation, normalises the rotation axis then calculates the angle of rotation. Uses 
    Rodrigues' rotation formula to find the rotation matrix for this. The difference vector is then rotated.

    Returns the rotated difference vector (array) and rotation matrix (array).

    """

    normalized_vector = difference_vector / np.linalg.norm(difference_vector)
    
    z_axis = np.array([0.0, 0.0, 1.0])
    
    # Calculate the axis of rotation by the cross product
    rotation_axis = np.cross(normalized_vector, z_axis)
    axis_norm = np.linalg.norm(rotation_axis)
    
    if axis_norm < 1e-10:

        # If the axis is too small, the vector is almost aligned with the z-axis
        rotation_matrix = np.eye(3)

    else:

        # Normalize the rotation axis
        rotation_axis /= axis_norm
        
        # Calculate the angle of rotation by the dot product
        cos_theta = np.dot(normalized_vector, z_axis)
        sin_theta = axis_norm
        
        # Rodrigues' rotation formula
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]], [rotation_axis[2], 0, -rotation_axis[0]], [-rotation_axis[1], rotation_axis[0], 0]])
        
        rotation_matrix = np.eye(3, dtype=np.float64) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    
    
    # Rotate the difference vector to align it with the z-axis
    difference_vector_rotated = np.dot(rotation_matrix, difference_vector)
    
    return difference_vector_rotated, rotation_matrix




def bohr_to_angstrom(length): 
    
    """
    
    Requires length (float).

    Converts the length from units of bohr to angstroms, then returns this (float).
    
    """
    
    return constants.bohr_radius_in_angstrom * length




def angstrom_to_bohr(length): 
    
    """
    
    Requires length (float).

    Converts the length from units of angstroms to bohr, then returns this (float).
    
    """
    
    return length / constants.bohr_radius_in_angstrom 




def one_dimension_to_three(coordinates): 
    
    """
    
    Requires one-dimensional coordinates (array).

    Returns a 3D coordinate array (array).

    """
    
    return np.array([[0, 0, coord] for coord in coordinates])





def three_dimensions_to_one(coordinates): 
    
    """
    
    Requires three-dimensional coordinates (array).

    Returns a 1D coordinate array (array).

    """
    
    return np.array([atom_coord[2] for atom_coord in coordinates])
    



def finish_calculation(calculation):

    """
    
    Requires calculation (Calculation).

    Determines the total time taken for a TUNA calculation, prints this to the console with a finale message, ends the program.
    
    """

    end_time = time.perf_counter()
    total_time = end_time - calculation.start_time

    # Prints the finale message
    log(colored(f"\n{calculation_types.get(calculation.calculation_type)} calculation in TUNA completed successfully in {total_time:.2f} seconds.  :)\n","white"), calculation, 1)
    sys.exit()



def calculate_centre_of_mass(masses, coordinates): 
    
    """
    
    Requires masses (list) and coordinates (array).

    Calculates and returns the centre of mass, using optimised numpy tensor operations (float).
    
    """
    
    return np.einsum("i,ij->", masses, coordinates,optimize=True) / np.sum(masses)



def print_trajectory(molecule, energy, coordinates):

    """
    
    Requires molecule (Molecule), energy (float), and coordinates (array).

    Prints out a trajectory from an optimisation or MD simulation to a file called "tuna-trajectory.xyz", then close the file.

    """


    atoms = molecule.atoms

    with open("tuna-trajectory.xyz", "a") as file:
        
        # Prints energy and atoms
        file.write(f"{len(atoms)}\n")
        file.write(f"Coordinates from TUNA calculation, E = {energy:.10f}\n")

        coordinates_output = bohr_to_angstrom(coordinates)

        # Prints coordinates
        for i in range(len(atoms)):

            file.write(f"  {atoms[i]}      {coordinates_output[i][0]:6f}      {coordinates_output[i][1]:6f}      {coordinates_output[i][2]:6f}\n")

    file.close()




def error(message): 

    """
    
    Requires message (string).

    Prints formatted error message, with sad face, in red, to the console. Then exits TUNA.
    
    """
    
    print(colored(f"\nERROR: {message}  :(\n","light_red"))
    sys.exit()




def warning(message, space=1): 
    
    """
    
    Requires message (string), optional number of preceding spaces (int).

    Prints formatted warning message to the console, in yellow.
    
    """
    
    print(colored(f"\n{" " * space}WARNING: {message}","light_yellow"))




def log(message, calculation, priority=1, end="\n"):

    """
    
    Requires message (string), calculation (Calculation), and optional priority (int) and string ending (string).

    Prints the message to the console if the priority is high enough for the print level. If priority is 1, the message
    will be printed. If priority is 2, the message is printed unless the terse print level is requested. If the priority
    is 3, the message will only be printed if additional print is requested.
    
    """

    if priority == 1: print(message, end=end)
    elif priority == 2 and not calculation.terse: print(message, end=end)
    elif priority == 3 and calculation.additional_print: print(message, end=end)
