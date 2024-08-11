import sys
import numpy as np

class primitive_gaussian():

    def __init__(self, alpha, coeff, coordinates):
        
        self.alpha = alpha
        self.coeff = coeff
        self.coordinates = coordinates
        self.N = (2.0 * alpha / np.pi) ** 0.75

class atomic_orbital():

    def __init__(self, pgs, coordinates):
        
        self.pgs = pgs
        self.coordinates = coordinates

class atom():
   
    def __init__(self, atomic_orbitals, coordinates, Z):
        
        self.atomic_orbitals = atomic_orbitals
        self.coordinates = coordinates
        self.Z = Z
        
class molecule():

    def __init__(self, atoms, coordinates, Z_list, basis_functions):
        
        self.atoms = atoms
        self.coordinates = coordinates
        self.Z_list = Z_list
        self.basis_functions = basis_functions

def generate_sto_3g_orbitals(atom,location):

    orbitals = []
    
    if atom == "H" or atom == "XH":
    
        orbitals = [[primitive_gaussian(3.425250914, 0.1543289673, location), 
        primitive_gaussian(0.6239137298, 0.5353281423, location), 
        primitive_gaussian(0.1688554040, 0.4446345422, location)]]
        
    elif atom == "HE" or atom == "XHE":
        
        orbitals = [[primitive_gaussian(0.6362421394E+01, 0.1543289673E+00, location), 
        primitive_gaussian(0.1158922999E+01, 0.5353281423E+00, location), 
        primitive_gaussian(0.3136497915E+00, 0.4446345422E+00, location)]]
        
    return orbitals


def generate_sto_6g_orbitals(atom,location):

    orbitals = []
    
    if atom == "H" or atom == "XH":
    
        orbitals = [[primitive_gaussian(0.3552322122E+02, 0.9163596281E-02, location), 
        primitive_gaussian(0.6513143725E+01, 0.4936149294E-01, location), 
        primitive_gaussian(0.1822142904E+01, 0.1685383049E+00, location),
        primitive_gaussian(0.6259552659E+00, 0.3705627997E+00, location), 
        primitive_gaussian(0.2430767471E+00, 0.4164915298E+00, location), 
        primitive_gaussian(0.1001124280E+00, 0.1303340841E+00, location)]]
        
    elif atom == "HE" or atom == "XHE":
    
        orbitals = [[primitive_gaussian(0.6598456824E+02, 0.9163596281E-02, location), 
        primitive_gaussian(0.1209819836E+02, 0.4936149294E-01, location), 
        primitive_gaussian(0.3384639924E+01, 0.1685383049E+00, location),
        primitive_gaussian(0.1162715163E+01, 0.3705627997E+00, location), 
        primitive_gaussian(0.4515163224E+00, 0.4164915298E+00, location), 
        primitive_gaussian(0.1859593559E+00, 0.1303340841E+00, location)]]
        
        
    return orbitals
    
    
def generate_6_31g_orbitals(atom,location):

    orbitals = []
    
    if atom == "H" or atom == "XH":
    
        orbitals = [[primitive_gaussian( 18.7311370000, 0.0334945995, location), 
        primitive_gaussian(2.8253937000, 0.2347269467, location), 
        primitive_gaussian(0.6401217000, 0.8137573184, location)],[primitive_gaussian( 0.1612778000, 1.000000, location)]]
    
    elif atom == "HE" or atom == "XHE":
    
        orbitals = [[primitive_gaussian(0.3842163400E+02, 0.0401397393, location), 
        primitive_gaussian(5.7780300000, 0.2612460970, location), 
        primitive_gaussian(1.2417740000, 0.7931846246, location)],[primitive_gaussian( 0.2979640000, 1.000000, location)]]
    
    
    return orbitals


def generate_6_31_plusg_orbitals(atom,location):

    orbitals = []
    
    if atom == "H" or atom == "XH":
    
        orbitals = [[primitive_gaussian( 18.7311370000, 0.0334945995, location), 
        primitive_gaussian(2.8253937000, 0.2347269467, location), 
        primitive_gaussian(0.6401217000, 0.8137573184, location)],[primitive_gaussian( 0.1612778000, 1.000000, location)]]
    
    elif atom == "HE" or atom == "XHE":
    
        orbitals = [[primitive_gaussian(0.3842163400E+02, 0.0401397393, location), 
        primitive_gaussian(5.7780300000, 0.2612460970, location), 
        primitive_gaussian(1.2417740000, 0.7931846246, location)],[primitive_gaussian( 0.2979640000, 1.000000, location)]]
    
    
    return orbitals


def generate_6_31_plus_plusg_orbitals(atom,location):

    orbitals = []
    
    if atom == "H" or atom == "XH":
    
        orbitals = [[primitive_gaussian(0.1873113696E+02, 0.3349460434E-01, location), 
        primitive_gaussian(0.2825394365E+01, 0.2347269535E+00, location), 
        primitive_gaussian(0.6401216923E+00, 0.8137573261E+00, location)],[primitive_gaussian(0.1612777588E+00, 1.000000, location)],[primitive_gaussian(0.3600000000E-01, 0.1000000000E+01, location)]]
    
    elif atom == "HE" or atom == "XHE":
    
        orbitals = [[primitive_gaussian(0.3842163400E+02, 0.4013973935E-01, location), 
        primitive_gaussian(0.5778030000E+01, 0.2612460970E+00, location), 
        primitive_gaussian(0.1241774000E+01, 0.7931846246E+00, location)],[primitive_gaussian(0.2979640000E+00, 1.000000, location)],[primitive_gaussian(0.8600000000E-01, 1.000000, location)]]
    
    
    return orbitals


def generate_3_21g_orbitals(atom, location):

    orbitals = []
    
    if atom == "H" or atom == "XH":
    
        orbitals = [[primitive_gaussian(0.5447178000E+01, 0.1562849787E+00, location), 
        primitive_gaussian(0.8245472400E+00, 0.9046908767E+00, location)],[primitive_gaussian(0.1831915800E+00, 1.000000, location)]]
    
    elif atom == "HE" or atom == "XHE":
    
        orbitals = [[primitive_gaussian(0.1362670000E+02, 0.1752298718E+00, location), 
        primitive_gaussian(0.1999350000E+01 , 0.8934823465E+00, location)],[primitive_gaussian(0.3829930000E+00, 1.000000, location)]]
    
    
    return orbitals
    
    
def generate_4_31g_orbitals(atom, location):

    orbitals = []
    
    if atom == "H" or atom == "XH":
    
        orbitals = [[primitive_gaussian(0.1873113696E+02, 0.3349460434E-01, location), 
        primitive_gaussian(0.2825394365E+01, 0.2347269535E+00, location),primitive_gaussian(0.6401216923E+00, 0.8137573261E+00, location)],
        [primitive_gaussian(0.1612777588E+00, 1.000000, location)]]
    
    elif atom == "HE" or atom == "XHE":
    
        orbitals = [[primitive_gaussian(0.3842163400E+02, 0.4013973935E-01, location), 
        primitive_gaussian(0.5778030000E+01, 0.2612460970E+00, location), 
        primitive_gaussian(0.1241774000E+01, 0.7931846246E+00, location)],[primitive_gaussian(0.2979640000E+00, 1.000000, location)]]
    
    
    return orbitals
    
    
def generate_6_311g_orbitals(atom,location):

    orbitals = []
    
    if atom == "H" or atom == "XH":
    
        orbitals = [[primitive_gaussian(33.86500, 0.0254938, location), 
        primitive_gaussian(5.094790, 0.190373, location), 
        primitive_gaussian(1.158790,  0.852161, location)],[primitive_gaussian(0.325840, 1.000000, location)],[primitive_gaussian(0.102741, 1.000000, location)]]
        
        
    elif atom == "HE" or atom == "XHE":
    
        orbitals = [[primitive_gaussian(98.12430, 0.0287452, location), 
        primitive_gaussian(14.76890, 0.208061, location), 
        primitive_gaussian(3.318830,  0.837635, location)],[primitive_gaussian(0.874047, 1.000000, location)],[primitive_gaussian(0.244564, 1.000000, location)]]
        
        
    return orbitals
    
def generate_6_311_plusg_orbitals(atom,location):

    orbitals = []
    
    if atom == "H" or atom == "XH":
    
        orbitals = [[primitive_gaussian(33.86500, 0.0254938, location), 
        primitive_gaussian(5.094790, 0.190373, location), 
        primitive_gaussian(1.158790,  0.852161, location)],[primitive_gaussian(0.325840, 1.000000, location)],[primitive_gaussian(0.102741, 1.000000, location)]]
        
        
    elif atom == "HE" or atom == "XHE":
    
        orbitals = [[primitive_gaussian(98.12430, 0.0287452, location), 
        primitive_gaussian(14.76890, 0.208061, location), 
        primitive_gaussian(3.318830,  0.837635, location)],[primitive_gaussian(0.874047, 1.000000, location)],[primitive_gaussian(0.244564, 1.000000, location)]]
        
        
    return orbitals
    
    
def generate_6_311_plus_plusg_orbitals(atom,location):

    orbitals = []

    if atom == "H" or atom == "XH":
    
        orbitals = [[primitive_gaussian(33.86500, 0.0254938, location), 
        primitive_gaussian(5.094790, 0.190373, location), 
        primitive_gaussian(1.158790, 0.852161, location)],[primitive_gaussian(0.325840, 1.000000, location)],[primitive_gaussian(0.102741, 1.000000, location)],[primitive_gaussian(0.036, 1.000000, location)]]
    
    if atom == "HE" or atom == "XHE": sys.exit("\nERROR: The 6-311++G is not parameterised for He. Choose another basis set!")
    
    return orbitals
    
    
def generate_hto_cbs_orbitals(atom,location):

    orbitals = []

    if atom == "H" or atom == "XH" or atom == "HE" or atom == "XHE":
    
        orbitals = [[primitive_gaussian(98.12430, 0.0287452, location), 
        primitive_gaussian(14.76890, 0.208061, location), 
        primitive_gaussian(3.318830,  0.837635, location)],[primitive_gaussian(0.874047, 1.000000, location)],[primitive_gaussian(0.244564, 1.000000, location)],[primitive_gaussian(0.036, 1.000000, location)],
        [primitive_gaussian(0.01, 1.000000, location)],[primitive_gaussian(30, 1.000000, location)],[primitive_gaussian(1000, 1.000000, location)], 
        [primitive_gaussian(35, 1.000000, location)], [primitive_gaussian(500, 1.000000, location)], [primitive_gaussian(100, 1.000000, location)],
        [primitive_gaussian(0.001, 1.000000, location)],[primitive_gaussian(33.86500, 0.0254938, location), 
        primitive_gaussian(5.094790, 0.190373, location), 
        primitive_gaussian(1.158790, 0.852161, location)],[primitive_gaussian(0.325840, 1.000000, location)],[primitive_gaussian(0.102741, 1.000000, location)]]
    
   
    
    return orbitals