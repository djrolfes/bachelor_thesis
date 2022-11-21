import numpy as np
from fibonacci import generate_vertices, generate_vertices_angles
from lattice_optimisation import lattice_optimizer, angles_to_cartesian

Start = 20
Stop = 150
Lattice_points = np.unique(np.geomspace(Start, Stop, 30, 1, dtype=int))
print(Lattice_points)

neighbors = 1
eps = 1e-13



def main():
    for p in Lattice_points:
        print(p)
        lattice = generate_vertices(p)
        #angles = generate_vertices_angles(p)
        #lattice_angles, _, _ = lattice_optimizer(angles, eps = eps, neighbors=neighbors) 
        # only goes up to 2**15 and consumes A LOT of memory
        #lattice = angles_to_cartesian(lattice_angles)
        np.savetxt(f"lattices\\phase\\fib_{p}.csv", lattice, delimiter="\t", newline="\n",fmt="%.13e")
    return

if __name__ == "__main__":
    main()