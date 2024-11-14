import numpy as np
from tuna_util import *

k = constants.k
c = constants.c
h = constants.h


def calculate_translational_internal_energy(temperature): 

    """
    
    Requires temperature (float).

    Calculates and returns the translational internal energy from statistical mechanics (float).
    
    """

    return (3 / 2) * k * temperature



def calculate_rotational_entropy(point_group, temperature, rotational_constant_per_m):
    
    """
    
    Requires point group (string), temperature (float), rotational constant (float).

    Calculates the rotational entropy, depending on the point group.

    Returns rotational entropy (float).
    
    """

    rotational_constant_per_bohr = bohr_to_angstrom(rotational_constant_per_m) * 1e-10

    if point_group == "Dinfh": symmetry_number = 2
    elif point_group == "Cinfv": symmetry_number = 1

    #Equation for rotational entropy from statistical mechanics
    rotational_entropy = k * (1 + np.log(k * temperature / (symmetry_number * rotational_constant_per_bohr * h * c)))

    return rotational_entropy




def calculate_rotational_internal_energy(temperature): 
    
    """
    
    Requires temperature (float).

    Calculates and returns rotational internal energy for a diatomic (float).

    """
    
    return k * temperature




def calculate_vibrational_internal_energy(frequency_per_cm, temperature): 
    
    """
    
    Requires frequency (float), temperature (float).

    Calculates and returns vibrational internal energy for a diatomic (float), using an equation from statistical mechanics.

    """
     
    vibrational_temperature = calculate_vibrational_temperature(frequency_per_cm)
    
    #Makes sure an error message isn't printed when dividing by a very small number
    with np.errstate(divide='ignore'):
        
        vibrational_internal_energy = k * vibrational_temperature / (np.exp(vibrational_temperature / temperature) - 1)


    return vibrational_internal_energy




def calculate_electronic_entropy(): 
    
    """
    
    Assumes an electronic entropy of zero for finite band gap materials, returns 0.

    """
    
    return 0




def calculate_vibrational_entropy(frequency_per_cm, temperature):

    """
    
    Requires frequency (float) and temperature (float).

    Returns vibrational entropy (float), calculated from statistical mechanics equation.
    
    """

    #Calculates the vibrational temperature
    vibrational_temperature = calculate_vibrational_temperature(frequency_per_cm)

    vibrational_entropy = k * (vibrational_temperature / (temperature * (np.exp(vibrational_temperature / temperature) - 1)) - np.log(1 - np.exp(-vibrational_temperature / temperature)))

    return vibrational_entropy




def calculate_translational_entropy(temperature, pressure, mass):

    """
    
    Requires temperature (float), pressure (float) and molecular mass (float).

    Returns translational entropy (float), from equation from statistical mechanics.
    
    """

    pressure_atomic_units = pressure / constants.pascal_in_atomic_units

    translational_entropy = k * (5 / 2 + np.log(((h * mass * k * temperature) / (h ** 2) ) ** (3/2) * (k * temperature / pressure_atomic_units)))

    return translational_entropy




def calculate_entropy(temperature, frequency_per_cm, point_group, rotational_constant_per_m, masses, pressure):

    """
    
    Requires temperature (float), frequency (float), point group (string), rotational constant (float), masses (array) and pressure (float).

    Calculates all the different contributions to entropy, then adds them all together.

    Returns entropy (float), translational entropy, rotational entropy, vibrational entropy and electronic entropy (floats).

    """

    translational_entropy = calculate_translational_entropy(temperature, pressure, np.sum(masses))
    rotational_entropy = calculate_rotational_entropy(point_group, temperature, rotational_constant_per_m)
    vibrational_entropy = calculate_vibrational_entropy(frequency_per_cm, temperature)
    electronic_entropy = calculate_electronic_entropy()

    #Total entropy is just the sum of all the contributions
    S = translational_entropy + rotational_entropy + vibrational_entropy + electronic_entropy

    return S, translational_entropy, rotational_entropy, vibrational_entropy, electronic_entropy




def calculate_vibrational_temperature(frequency_per_cm):

    """
    
    Requires frequency (float).
    
    Calculates and returns vibrational temperature, from statistical mechanics equation (float).

    """

    #Conversion into atomic units
    frequency_per_bohr = bohr_to_angstrom(frequency_per_cm) * 1e-8

    vibrational_temperature = h * frequency_per_bohr * c / k

    return vibrational_temperature




def calculate_internal_energy(E, E_ZPE, temperature, frequency_per_cm):

    """
    
    Requires energy (float), zero-point energy (float), temperature (float) and frequency (float).

    Calculates all contributions to internal energy, and adds them together to the total internal energy.

    Returns internal energy (float), translational, rotational and vibrational contributions (floats).
    
    """

    translational_internal_energy = calculate_translational_internal_energy(temperature)
    rotational_internal_energy = calculate_rotational_internal_energy(temperature)
    vibrational_internal_energy = calculate_vibrational_internal_energy(frequency_per_cm, temperature)

    #Adds together all contributions to internal energy
    U = E + E_ZPE + translational_internal_energy + rotational_internal_energy + vibrational_internal_energy

    return U, translational_internal_energy, rotational_internal_energy, vibrational_internal_energy



def calculate_enthalpy(U, temperature): 
    
    """
    
    Requires internal energy (float) and temperature (float).

    Returns enthalpy (float).

    """

    return U + k * temperature




def calculate_free_energy(H, temperature, S):
        
    """
    
    Requires enthalpy (float), temperature (float) and entropy (float).

    Returns Gibbs free energy (float).

    """
    
    return H - temperature * S