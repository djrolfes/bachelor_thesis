import numpy as np
from fibonacci import generate_vertices, generate_vertices_angles
from lattice_optimisation import lattice_optimizer, angles_to_cartesian

N = 21
Lattice_points = [i for i in range(15,N)]
neighbors = 1
eps = 1e-13



def main():
    for p in Lattice_points:
        print(p)
        angles = generate_vertices_angles(2**p)
        print(angles)
        lattice_angles, _, _ = lattice_optimizer(angles, eps = eps, neighbors=neighbors) 
        # only goes up to 2**15 an consumes A LOT of memory
        print(lattice_angles)
        lattice = angles_to_cartesian(lattice_angles)
        np.savetxt(f"lattices\optim_{2**p}.csv", lattice, delimiter="\t", newline="\n",fmt="%.13e")
    return

if __name__ == "__main__":
    main()