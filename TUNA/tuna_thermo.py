import numpy as np
import tuna_util as util


k = util.constants.k
c = util.constants.c
h = util.constants.h


def calculate_translational_internal_energy(temperature): 

    return (3 / 2) * k * temperature


def calculate_rotational_entropy(point_group, temperature, rotational_constant_per_m):
    
    rotational_constant_per_bohr = util.bohr_to_angstrom(rotational_constant_per_m) * 1e-10

    if point_group == "Dinfh": symmetry_number = 2
    elif point_group == "Cinfv": symmetry_number = 1


    rotational_entropy = k * (1 + np.log(k * temperature / (symmetry_number * rotational_constant_per_bohr * h * c)))

    return rotational_entropy


def calculate_rotational_internal_energy(temperature): return k * temperature


def calculate_vibrational_internal_energy(frequency_per_cm, temperature): 
    
    vibrational_temperature = calculate_vibrational_temperature(frequency_per_cm)
    
    with np.errstate(divide='ignore'):
        
        vibrational_internal_energy = k * vibrational_temperature / (np.exp(vibrational_temperature / temperature) - 1)


    return vibrational_internal_energy


def calculate_electronic_entropy(): return 0


def calculate_vibrational_entropy(frequency_per_cm, temperature):

    vibrational_temperature = calculate_vibrational_temperature(frequency_per_cm)

    S = k * (vibrational_temperature / (temperature * (np.exp(vibrational_temperature / temperature) - 1)) - np.log(1 - np.exp(-vibrational_temperature / temperature)))

    return S


def calculate_translational_entropy(temperature, pressure, mass):

    pressure_atomic_units = pressure / util.constants.pascal_in_atomic_units

    translational_entropy = k * (5 / 2 + np.log(((h * mass * k * temperature) / (h ** 2) ) ** (3/2) * (k * temperature / pressure_atomic_units)))

    return translational_entropy



def calculate_entropy(temperature, frequency_per_cm, point_group, rotational_constant_per_m, masses, pressure):

    translational_entropy = calculate_translational_entropy(temperature, pressure, np.sum(masses))
    rotational_entropy = calculate_rotational_entropy(point_group, temperature, rotational_constant_per_m)
    vibrational_entropy = calculate_vibrational_entropy(frequency_per_cm, temperature)
    electronic_entropy = calculate_electronic_entropy()

    S = translational_entropy + rotational_entropy + vibrational_entropy + electronic_entropy


    return S, translational_entropy, rotational_entropy, vibrational_entropy, electronic_entropy



def calculate_vibrational_temperature(frequency_per_cm):

    frequency_per_bohr = util.bohr_to_angstrom(frequency_per_cm) * 1e-8

    vibrational_temperature = h * frequency_per_bohr * c / k

    return vibrational_temperature



def calculate_internal_energy(E, E_ZPE, temperature, frequency_per_cm):

    translational_internal_energy = calculate_translational_internal_energy(temperature)
    rotational_internal_energy = calculate_rotational_internal_energy(temperature)
    vibrational_internal_energy = calculate_vibrational_internal_energy(frequency_per_cm, temperature)

    U = E + E_ZPE + translational_internal_energy + rotational_internal_energy + vibrational_internal_energy


    return U, translational_internal_energy, rotational_internal_energy, vibrational_internal_energy


def calculate_enthalpy(U, temperature): return U + k * temperature

def calculate_free_energy(H, temperature, S): return H - temperature * S