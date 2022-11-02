from genericpath import exists
from su2McExecutor import executor
import numpy as np
import pathlib

# Create directory for data
pathlib.Path("tmp/scan").mkdir(parents=True, exist_ok=True)

# Create executor Object. Paramater is number of threads!
ex = executor(8)


betas =np.linspace(0.1, 10, 100)

# Collect some data
ex.recordCPUData(
    # 8^4 lattice
    8,
    # coupling constants
    betas=np.linspace(0.1, 10, 20),
    # partition type
    partition="--partition-linear-weighted",
    # additional parameter for partition
    partitionOpt="5",
    # directory to store results in
    dataDir="tmp/scan",
    # cold or hot start
    cold=True,
    # number of iterations to run (8000 is a safe value, smaller probably fine as well)
    sweeps=100
)

ex.runEvaluator(
    # directory which to evaluate
    "tmp/scan",
    # where to save results
    "tmp/scan.csv",
    # how many of the initial iterations
    # to throw away (5000 is a safe value, smaller probably fine as well)
    50)

plaquette = np.genfromtxt("tmp/scan.csv")[:,1]
plaquette_std_dev = np.genfromtxt("tmp/scan.csv")[:,2]


print(plaquette)