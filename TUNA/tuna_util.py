import numpy as np
import time, sys
import tuna_basis as basis_sets
from termcolor import colored

calculation_types = {"SPE": "Single point energy", "OPT": "Geometry optimisation", "FREQ": "Harmonic frequency", "OPTFREQ": "Optimisation and harmonic frequency", "SCAN": "Coordinate scan", "MD": "Ab initio molecular dynamics", "ANHARM": "Anharmonic frequency"}
method_types = {"HF": "Hartree-Fock theory", "RHF": "restricted Hartree-Fock theory", "RHF": "unrestricted Hartree-Fock theory", "MP2": "MP2 theory", "UMP2": "unrestricted MP2 theory", "SCS-MP2": "spin-component-scaled MP2 theory", "MP3": "MP3 theory", "UMP3": "unrestricted MP3 theory", "SCS-MP3": "spin-component-scaled MP3 theory"}


class Constants:

    def __init__(self):

        self.planck_constant_in_joules_seconds = 6.62607015e-34
        self.elementary_charge_in_coulombs = 1.602176634e-19
        self.electron_mass_in_kilograms = 9.1093837139e-31
        self.permittivity_in_farad_per_metre = 1.11265005620e-10

        self.c_in_metres_per_second = 299792458
        self.k_in_joules_per_kelvin = 1.380649e-23
        self.atomic_mass_unit_in_kg = 1.660539068911e-27

        self.reduced_planck_constant_in_joules_seconds = self.planck_constant_in_joules_seconds / (2 * np.pi)

        self.bohr_in_metres = self.permittivity_in_farad_per_metre * self.reduced_planck_constant_in_joules_seconds ** 2 / (self.electron_mass_in_kilograms * self.elementary_charge_in_coulombs ** 2)
        self.hartree_in_joules = self.reduced_planck_constant_in_joules_seconds ** 2 / (self.electron_mass_in_kilograms * self.bohr_in_metres ** 2)
        self.atomic_time_in_seconds = self.reduced_planck_constant_in_joules_seconds /  self.hartree_in_joules
        self.atomic_time_in_femtoseconds = self.atomic_time_in_seconds * 10 ** 15
        self.bohr_radius_in_angstrom = self.bohr_in_metres * 10 ** 10

        self.pascal_in_atomic_units = self.hartree_in_joules / self.bohr_in_metres ** 3

        self.per_cm_in_hartree = self.hartree_in_joules / (self.c_in_metres_per_second * self.planck_constant_in_joules_seconds * 10 ** 2)
        self.per_cm_in_GHz = self.hartree_in_joules / (self.planck_constant_in_joules_seconds * self.per_cm_in_hartree * 10 ** 9)

        self.c = self.c_in_metres_per_second * self.atomic_time_in_seconds / self.bohr_in_metres
        self.k = self.k_in_joules_per_kelvin / self.hartree_in_joules
        self.h = self.planck_constant_in_joules_seconds / (self.hartree_in_joules * self.atomic_time_in_seconds)

        self.atomic_mass_unit_in_electron_mass = self.atomic_mass_unit_in_kg / self.electron_mass_in_kilograms

        self.atom_masses_in_amu = {"H": 1.00782503223, "HE": 4.00260325413}
        self.atom_masses = {"H": self.atom_masses_in_amu.get("H") * self.atomic_mass_unit_in_electron_mass, "HE": self.atom_masses_in_amu.get("HE") * self.atomic_mass_unit_in_electron_mass}



constants = Constants()

def bohr_to_angstrom(length): return constants.bohr_radius_in_angstrom * length

def angstrom_to_bohr(length): return length / constants.bohr_radius_in_angstrom 

def one_dimension_to_three(coordinates): return np.array([[0, 0, coord] for coord in coordinates])

def three_dimensions_to_one(coordinates): return np.array([atom_coord[2] for atom_coord in coordinates])
    


def finish_calculation(calculation):

    end_time = time.perf_counter()
    total_time = end_time - calculation.start_time

    print(colored(f"\n{calculation_types.get(calculation.calculation_type)} calculation in TUNA completed successfully in {total_time:.2f} seconds.  :)\n","white"))
    sys.exit()




class Calculation:

    loose = {"delta_E": 0.000001, "maxDP": 0.00001, "rmsDP": 0.000001, "orbitalGrad": 0.0001, "word": "loose"}
    normal = {"delta_E": 0.0000001, "maxDP": 0.000001, "rmsDP": 0.0000001, "orbitalGrad": 0.00001, "word": "medium"}
    tight = {"delta_E": 0.000000001, "maxDP": 0.00000001, "rmsDP": 0.000000001, "orbitalGrad": 0.0000001, "word": "tight"}
    extreme = {"delta_E": 0.00000000001, "maxDP": 0.0000000001, "rmsDP": 0.00000000001, "orbitalGrad": 0.000000001, "word": "extreme"}   

    looseopt = {"gradient": 0.001, "step": 0.01, "word": "loose"}
    normalopt = {"gradient": 0.0001, "step": 0.0001, "word": "medium"}
    tightopt = {"gradient": 0.000001, "step": 0.00001, "word": "tight"}
    extremeopt = {"gradient": 0.00000001, "step": 0.0000001, "word": "extreme"}



    def __init__(self, calculation_type, method, start_time, params, basis):


        self.calculation_type = calculation_type
        self.method = method
        self.start_time = start_time
        self.basis = basis
        self.mp2_type = None
        
        self.additional_print = False
        self.terse = False
        self.charge = 0
        self.multiplicity = 1
        self.multiplicity_defined = False
        self.norotate_guess = False
        self.rotate_guess = False
        
        if self.calculation_type == "OPT" or self.calculation_type == "FREQ" or self.calculation_type == "OPTFREQ" or self.calculation_type == "MD": self.scf_conv = self.tight
        else: self.scf_conv = self.normal

        self.slowconv = False
        self.veryslowconv = False
        self.diis = True
        self.level_shift = False
        self.damping = True
        self.max_iter = 50
        self.level_shift_parameter = 0.2
        
        self.scanstep = None
        self.scannumber = None
        self.scanplot = False
        self.densplot = False

        self.md_number_of_steps = 50
        self.timestep = 0.1
        
        self.moread = True  
        self.nomoread = False  
        self.geom_conv = self.tightopt
        self.calchess = False
        self.optmax = False
        self.trajectory = False
        self.notrajectory = False
        self.d2 = False
        self.geom_max_iter = 20
        self.decontract = False
        self.theta = np.pi / 4

        if self.calculation_type == "MD": self.temperature = 0
        else: self.temperature = 298.15

        self.pressure = 101325

        self.process_params(params)


    def process_params(self, params):
        
        if "NODIIS" in params: self.diis = False 
        if "DECONTRACT" in params: self.decontract = True 
        if "DAMP" in params: self.damping = True
        elif "NODAMP" in params: self.damping = False
        if "DENSPLOT" in params: self.densplot = True
        if "SCANPLOT" in params: self.scanplot = True
        if "LEVELSHIFT" in params: self.level_shift = True
        if "SCANPLOT" in params: self.scan_plot = True
        if "D2" in params: self.d2 = True
        if "CALCHESS" in params: self.calchess = True
        if "P" in params: self.additional_print = True
        elif "T" in params: self.terse = True
        if "NOMOREAD" in params: self.nomoread = True
        if "OPTMAX" in params: self.optmax = True
        if "TRAJ" in params: self.trajectory = True
        elif "NOTRAJ" in params: self.notrajectory = True
        if "ROTATE" in params: self.rotate_guess = True
        elif "NOROTATE" in params: self.norotate_guess = True

        if "SLOWCONV" in params: 
            
            self.damping = True
            self.slowconv = True

        elif "VERYSLOWCONV" in params: 
            
            self.damping = True
            self.veryslowconv = True


        if "LOOSE" in params or "LOOSESCF" in params: self.scf_conv = self.loose  
        elif "NORMAL" in params or "NORMALSCF" in params: self.scf_conv = self.normal  
        elif "TIGHT" in params or "TIGHTSCF" in params: self.scf_conv = self.tight   
        elif "EXTREME" in params or "EXTREMESCF" in params: self.scf_conv = self.extreme 

        if "LOOSEOPT" in params: self.geom_conv = self.looseopt  
        elif "NORMALOPT" in params: self.geom_conv = self.normalopt  
        elif "TIGHTOPT" in params: self.geom_conv = self.tightopt 
        elif "EXTREMEOPT" in params: self.geom_conv = self.extremeopt    


        def get_param_value(param_name, value_type):

            if param_name in params:

                try: return value_type(params[params.index(param_name) + 1])
                except IndexError: error(f"Parameter \"{param_name}\" requested but no value specified!")
                except ValueError: error(f"ERROR: Parameter \"{param_name}\" must be of type {value_type.__name__}!")
            
            return 
        

        self.charge = get_param_value("CHARGE", int) if get_param_value("CHARGE", int) is not None else self.charge
        self.charge = get_param_value("CH", int) if get_param_value("CH", int) is not None else self.charge 
        self.multiplicity = get_param_value("MULTIPLICITY", int) if get_param_value("MULTIPLICITY", int) is not None else self.multiplicity
        self.multiplicity = get_param_value("ML", int) if get_param_value("ML", int) is not None else self.multiplicity

        self.max_iter = get_param_value("MAXITER", int) or self.max_iter
        self.geom_max_iter = get_param_value("GEOMMAXITER", int) or self.geom_max_iter
        self.geom_max_iter = get_param_value("MAXGEOMITER", int) or self.geom_max_iter
        self.scanstep = get_param_value("SCANSTEP", float) or self.scanstep
        self.scannumber = get_param_value("SCANNUMBER", int) or self.scannumber
        self.md_number_of_steps = get_param_value("MDNUMBER", int) or self.md_number_of_steps
        self.timestep = get_param_value("TIMESTEP", float) or self.timestep

        self.temperature = get_param_value("TEMP", float) or self.temperature
        self.temperature = get_param_value("TEMPERATURE", float) or self.temperature
        self.pressure = get_param_value("PRES", float) or self.pressure
        self.pressure = get_param_value("PRESSURE", float) or self.pressure

        self.theta = get_param_value("THETA", float) * np.pi / 180 if get_param_value("THETA", float) else self.theta
        
        self.default_multiplicity = False if "ML" in params or "MULTIPLICITY" in params else True


class Molecule:

    def __init__(self, atoms, coordinates, calculation):

        atom_charges = {"XH": 0, "XHE": 0, "H": 1, "HE": 2}

        atom_masses = Constants().atom_masses

        self.atoms = atoms
        self.coordinates = coordinates
        self.charge = calculation.charge
        self.multiplicity = calculation.multiplicity
        self.basis = calculation.basis

        self.masses = np.array([atom_masses[atom] for atom in self.atoms if "X" not in atom])
        self.Z_list = [atom_charges[atom] for atom in self.atoms]

        self.n_electrons = np.sum(self.Z_list) - self.charge



        self.point_group = self.determine_point_group()
        self.molecular_structure = self.determine_molecular_structure()

        self.mol = [basis_sets.generate_atomic_orbitals(atom, self.basis, coord) for atom, coord in zip(self.atoms, self.coordinates)]    
        self.ao_ranges = [len(basis_sets.generate_atomic_orbitals(atom, self.basis, coord)) for atom, coord in zip(self.atoms, self.coordinates)]
        self.atomic_orbitals = [orbital for atom_orbitals in self.mol for orbital in atom_orbitals]  
        self.pgs = [pg for atomic_orbital in self.atomic_orbitals for pg in atomic_orbital]

        if len(self.atoms) == 2: self.bond_length = np.linalg.norm(coordinates[1] - coordinates[0])


        if calculation.default_multiplicity and self.n_electrons % 2 != 0: self.multiplicity = 2



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

        if calculation.method in ["MP3", "RMP3", "UMP3"]: calculation.mp2_basis = "SO" 
    



        self.n_unpaired_electrons = self.multiplicity - 1
        self.n_alpha = int((self.n_electrons + self.n_unpaired_electrons) / 2)
        self.n_beta = int(self.n_electrons - self.n_alpha)
        self.n_doubly_occ = min(self.n_alpha, self.n_beta)
        self.n_occ = self.n_alpha + self.n_beta


        if self.n_electrons % 2 == 0 and self.multiplicity % 2 == 0: error("Impossible charge and multiplicity combination (both even)!")
        if self.n_electrons % 2 != 0 and self.multiplicity % 2 != 0: error("Impossible charge and multiplicity combination (both odd)!")
        if self.n_electrons - self.multiplicity < -1: error("Multiplicity too high for number of electrons!")
        if self.multiplicity < 1: error("Multiplicity must be at least 1!")


        calculation.n_electrons_per_orbital = 2 if calculation.reference == "RHF" else 1





    def determine_point_group(self):
    
        if len(self.atoms) == 2 and "XH" not in self.atoms and "XHE" not in self.atoms:
       
            return "Dinfh" if self.atoms[0] == self.atoms[1] else "Cinfv"
            
        return "K"


    def determine_molecular_structure(self):
    
        if len(self.atoms) == 2 and "XH" not in self.atoms and "XHE" not in self.atoms:
            
            return f"{self.atoms[0].lower().capitalize()} --- {self.atoms[1].lower().capitalize()}"
            
        return self.atoms[0].lower().capitalize()



class Output:

    def __init__(self, energy, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta, epsilons, epsilons_alpha, epsilons_beta):

        self.energy = energy

        self.P = P
        self.S = S


        self.P_alpha = P_alpha
        self.P_beta = P_beta

        self.molecular_orbitals = molecular_orbitals
        self.molecular_orbitals_alpha = molecular_orbitals_alpha
        self.molecular_orbitals_beta = molecular_orbitals_beta

        self.epsilons = epsilons
        self.epsilons_alpha = epsilons_alpha
        self.epsilons_beta = epsilons_beta

        self.epsilons_combined = np.append(self.epsilons_alpha, self.epsilons_beta)




def rotate_coordinates_to_z_axis(difference_vector):

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



def error(message): 
    
    print(colored(f"\nERROR: {message}  :(\n","light_red"))
    sys.exit()

def warning(message, space=1): print(colored(f"\n{" " * space}WARNING: {message}","light_yellow"))


def big_log(message, calculation, end="\n"):

    if calculation.additional_print: print(message, end=end)


def log(message, calculation, end="\n"):

    if not calculation.terse: print(message, end=end)