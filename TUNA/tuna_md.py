import numpy as np

positions_old = [0,0]

masses = [1,1]
timestep = 0.1
number_of_steps = 100
positions_initial = [0,1]
temperature = 278
k = 1.381e-23

def get_initial_velocities(temperature,masses):

    velocities = np.random.normal(0, np.sqrt(k * temperature / masses),1)

    com_velocity = (masses[0] * velocity_1 + masses[1] * velocity_2) / total_mass
    velocities_adjusted = velocities - com_velocity

    return velocities_adjusted


def calculate_accelerations(forces, masses): return forces / masses
    

def velocity_verlet_integration(positions, velocities, accelerations, timestep): 
    
    positions_new = positions + velocities * timestep + 0.5 * accelerations * timestep ** 2
    accelerations_new = calculate_accelerations(forces, masses)
    velocities_new = velocities + 0.5 * (accelerations + accelerations_new) * timestep
    
    
def run_md(number_of_steps, masses, positions_initial, timestep):

    positions = positions_initial
    time = 0

    for i in range(number_of_steps):
      
        forces = calculate_forces(positions)
        calculate_accelerations(forces, masses)
        
        positions_old = positions        
        positions = verlet_integration(positions, positions_old, accelerations, timestep)
        
        time += timestep
        
        print(f"{time}    {positions[1]-positions[0]}")
        
        
run_md(number_of_steps, masses, positions_initial, timestep)