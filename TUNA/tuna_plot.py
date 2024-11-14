import sys        
from tuna_util import *

def scan_plot(calculation, bond_lengths, energies):

    """
    
    Requires calculation (Calculation), bond lengths and energies (lists).

    Uses matplotlib to plot and show a nicely formatted energy curve as a function of bond length.

    """

    log("Plotting energy profile diagram...   ", calculation, 1, end=""); sys.stdout.flush()
    
    import matplotlib.pyplot as plt
    import matplotlib

    #Various fixed parameters for making matplotlib graph for coordinate scan
    matplotlib.rcParams['font.family'] = 'Arial'
    _, ax = plt.subplots(figsize=(10,5))    
    plt.plot(bond_lengths, energies, color=(0.75,0,0),linewidth=1.75)
    plt.xlabel("Bond Length (Angstrom)", fontweight="bold", labelpad=10, fontfamily="Arial",fontsize=12)
    plt.ylabel("Energy (hartree)",labelpad=10, fontweight="bold", fontfamily="Arial",fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.25, length=6, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=11, width=1.25, length=3, direction='out')
    
    for spine in ax.spines.values(): spine.set_linewidth(1.25)
    
    plt.minorticks_on()
    plt.tight_layout() 
    log("[Done]", calculation, 1)
    
    #Shows the coordinate scan plot
    plt.show()


 
def construct_electron_density(P, grid_density, molecule, calculation):

    """
    
    Requires electron density (array), grid density (float), molecule (Molecule) and calculation (Calculation).

    Builds three-dimensional cube using molecular coordinates. Then determines array of atomic orbitals and contracts these with the density matrix
    to form the electron density, as a 3D grid. *Only works for s orbitals*. Plots fixed isosurface of the electron density.

    Returns three-dimensional electron density cube.
    
    """


    log("\n Beginning electron density surface plot calculation...\n", calculation, 1)
    log(" Setting up grid...   ", calculation, 1, end=""); sys.stdout.flush()
    
    coordinates = [molecule.coordinates[0][2], molecule.coordinates[1][2]]
    start = coordinates[0] - 4
    
    #Builds x,y and z axes for meshgrid
    x = np.arange(start, coordinates[0] + 4 + grid_density, grid_density)
    y = np.arange(start, coordinates[0] + 4 + grid_density, grid_density)
    z = np.arange(start, coordinates[1] + 4 + grid_density, grid_density)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    
    log("[Done]", calculation, 1)
    
    log(" Generating electron density cube...   ", calculation, 1, end=""); sys.stdout.flush()
    
    n = 0

    atomic_orbitals = []

    #Builds atomic orbitals 3D array for s orbitals only
    for orbital in molecule.atomic_orbitals:

        a = 0

        for pg in orbital:  
        
            a += pg.N * pg.coeff * np.exp(-pg.alpha * ((X - pg.coordinates[0])**2 + (Y - pg.coordinates[1])**2 + (Z - pg.coordinates[2])**2))
        
        
        atomic_orbitals.append(a)
    
    atomic_orbitals = np.array(atomic_orbitals)
    
    #Contracts density matrix with atomic orbital arrays to find electron density
    n = np.einsum("mn,mijk,nijk->ijk", P, atomic_orbitals, atomic_orbitals)
    
    #Normalises electron density to account for numerical grid-related loss of accuracy
    normalisation = np.trapz(np.trapz(np.trapz(n,z),y), x)
    n *= molecule.n_doubly_occ * 2 / normalisation

    log("[Done]", calculation, 1)
    log(" Generating surface plot...   ", calculation, 1, end="")
    sys.stdout.flush()
    isovalue = 0.06
    
    #Plots isosurface at fixed value, and projects this on the screen
    from skimage import measure
    import plotly.graph_objects as go
    
    verts, faces, _, _ = measure.marching_cubes(n, isovalue, spacing=(grid_density, grid_density, grid_density))
    intensity = np.full(len(verts), isovalue)
    
    fig = go.Figure(data=[go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],intensity=intensity,colorscale='Agsunset',opacity=0.5)])
    fig.update_layout(scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),bgcolor='rgb(255, 255, 255)'),margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(scene_camera=dict(eye=dict(x=0.5, y=2.5, z=0.5)))
    log("[Done]\n", calculation, 1)
    
    #Shows the figure in default browser
    fig.show()
    
    return n
