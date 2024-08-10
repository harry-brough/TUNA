import numpy as np
import matplotlib.pyplot as plt


#get potential energy surface and values
#do regression to get potential energy surface as taylor expansion of polynomials
#rebuild potential energy surface in high dimension (check with matplotlib)
#build Hamiltonian matrix
#diagonalise Hamiltonian matrix
#calculate difference between two smallest eigenvalues


def get_grid_size(): return 1000


def get_polynomial_degree(): return 20




def build_taylor_approximation(potential_energy_surface, positions, grid_size, polynomial_degree):

    coefficients = np.polyfit(positions, potential_energy_surface, polynomial_degree)

    positions_interpolated = np.linspace(min(positions), max(positions), grid_size)
    
    potential_energy_surface_interpolated = np.polyval(coefficients, positions_interpolated)


    return potential_energy_surface_interpolated, positions_interpolated



def build_nuclear_hamiltonian(potential_energy_surface, positions, grid_size):

    dr = positions[1] - positions[0]
    k = 1 / (2 * dr ** 2)

    T = k * (np.diag(2 * np.ones(grid_size)) - np.diag(np.ones(grid_size - 1), 1) - np.diag(np.ones(grid_size - 1), -1))
    U = np.diag(potential_energy_surface)
    V = 0.391/2 * (positions - 1.3936729166) ** 2
    print(V)
    plt.plot(positions, V)
    V = np.diag(V)
    H = T + V

    return H


def diagonalise_nuclear_hamiltonian(H):
    
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    eigenvalues = np.sort(eigenvalues)

    return eigenvalues, eigenvectors




def calculate_anharmonic_frequency(potential_energy_surface, positions):
    
    grid_size = get_grid_size()
    polynomial_degree = get_polynomial_degree()

    potential_energy_surface_interpolated, positions_interpolated = build_taylor_approximation(potential_energy_surface, positions, grid_size, polynomial_degree)

    H = build_nuclear_hamiltonian(potential_energy_surface_interpolated, positions_interpolated, grid_size)

    eigenvalues, eigenvectors = diagonalise_nuclear_hamiltonian(H)
    plt.plot(positions_interpolated,potential_energy_surface_interpolated+1)
    plt.plot(positions_interpolated,eigenvectors[:,0])
    plt.plot(positions_interpolated,eigenvectors[:,1])
    plt.plot(positions_interpolated,eigenvectors[:,2])
    plt.xlim(0,3)
    plt.ylim(-1,1)
    plt.show()
    sorted_eigenvalues = np.sort(eigenvalues)

    fundamental_absorbance = sorted_eigenvalues[1] - sorted_eigenvalues[0]

    return fundamental_absorbance


