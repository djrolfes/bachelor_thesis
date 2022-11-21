import numpy as np
import pathlib
from genericpath import exists
from su2McExecutor import executor




PATH = pathlib.Path("/mnt/f/Studium_Physik/Bachelorarbeit/code/bachelor_thesis/lattices")
paths = []
for i in range(14):
    paths.extend(list(PATH.absolute().glob(f"**/*_{int(2**i)}.csv")))

ex = executor(10)
betas =np.linspace(0.1, 10, 100)


for path in paths:
    print(path)
    print(path.parts[-1])
    pathlib.Path(f"tmp/{path.parts[-1][:-4]}").mkdir(parents=True, exist_ok=True)
    ## Collect some data
    ex.recordCPUData(
        # 8^4 lattice
        8,
        # coupling constants
        betas=betas,
        # partition type
        partition="--partition-list",
        # additional parameter for partition
        partitionOpt=path,
        # directory to store results in
        dataDir=f"tmp/{path.parts[-1][:-4]}",
        # cold or hot start
        cold=True,
        # number of iterations to run (8000 is a safe value, smaller probably fine as well)
        sweeps=8000
    )
    ex.runEvaluator(
        # directory which to evaluate
        f"tmp/{path.parts[-1][:-4]}",
        # where to save results
        f"tmp/{path.parts[-1][:-4]}.csv",
        # how many of the initial iterations
        # to throw away (5000 is a safe value, smaller probably fine as well)
        5000)

