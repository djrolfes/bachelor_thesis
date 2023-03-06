import multiprocessing as mp
import numpy as np
from lattice_optimisation import angles_to_cartesian, lattice_optimizer_scipy, lattice_optimizer, cartesian_to_angles, calc_mean_distance_angles_simple
from fibonacci import generate_vertices_angles
from create_lattice import simple_generate_vertices

fn = "lattices\\better_optimistion06\\means.txt"
num = 120
sizes = np.geomspace(2**4,2**20, num, endpoint=True, dtype=np.int64)
SEED = 152555489
SEED = 41159646512316855241152

def worker(arg, q):
    angles = cartesian_to_angles(simple_generate_vertices(arg, seed = SEED))
    opt_angles, mean = lattice_optimizer_scipy(angles)
    _, std = calc_mean_distance_angles_simple(opt_angles, std = True)
    opt_lattice = angles_to_cartesian(opt_angles)
    q.put(f"{arg}   {mean}  {std}")
    np.save(f"lattices\\better_optimistion06\\scipy_opt{arg}", opt_lattice)
    return arg

def listener(q):
    with open(fn, 'w') as f:
        while True:
            m = q.get()
            if m == 'kill':
                #f.write('killed')
                break
            f.write(m + "\n")
            f.flush()

def main():
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    watcher = pool.apply_async(listener, (q,))

    jobs = []
    for size in sizes:
        job = pool.apply_async(worker, (size, q))
        jobs.append(job)
    
    for job in jobs:
        message = job.get()
        print(f"Size {message} done")

    q.put('kill')
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()