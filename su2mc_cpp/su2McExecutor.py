import os, shutil, subprocess, pathlib
import numpy as np

from pathos.pools import ProcessPool


class executor:
    def __init__(self, threads):
        self.threads = threads

    def fixParameters(self, parameters):
        runCount = 1
        for j in range(len(parameters)):
            if isinstance(parameters[j], list) or isinstance(parameters[j], np.ndarray):
                runCount = len(parameters[j])

        out = [[] for i in parameters]

        for j in range(len(parameters)):
            if isinstance(parameters[j], list) or isinstance(parameters[j], np.ndarray):
                out[j] = parameters[j]
            else:
                out[j] = [parameters[j] for i in range(runCount)]
        return out

    def recordCPUData(
        self,
        latSize,
        betas,
        sweeps,
        dataDir,
        deltas=0.1,
        partition=None,
        partitionOpt=None,
        hits=None,
        multiSweep=None,
        cold=False,
        dimensions=4,
    ):

        (
            latSize,
            betas,
            deltas,
            sweeps,
            partition,
            partitionOpt,
            hits,
            cold,
            multiSweep,
            dimensions,
        ) = self.fixParameters(
            [
                latSize,
                betas,
                deltas,
                sweeps,
                partition,
                partitionOpt,
                hits,
                cold,
                multiSweep,
                dimensions,
            ]
        )

        if os.path.exists(dataDir):
            shutil.rmtree(dataDir)
        pathlib.Path(dataDir).mkdir(exist_ok=True, parents=True)

        pool = ProcessPool(nodes=self.threads)

        def threadFunc(i):
            callList = [
                "./main",
                "-m",
                str(sweeps[i]),
                "-b",
                str(betas[i]),
                "-d",
                str(deltas[i]),
                "-l",
                str(latSize[i]),
                "-o",
                dataDir + "/data-{}.csv".format(i),
                "--dimensions",
                str(dimensions[i]),
            ]

            if partition[i] != None:
                callList.append(partition[i])
                if partitionOpt[i] != None:
                    callList.append(partitionOpt[i])

            if hits[i] != None:
                callList.append("--hits")
                callList.append(str(hits[i]))
            if cold[i]:
                callList.append("-c")
            if multiSweep[i] != None:
                callList.append("--multi-sweep")
                callList.append(str(multiSweep[i]))

            subprocess.check_call(callList, stdout=subprocess.DEVNULL)

            print("Collecting CPU Dataset ({}/{}).".format(i + 1, len(betas)))

        results = pool.map(threadFunc, [i for i in range(len(betas))])

    def runEvaluator(self, dataDir, outputFile, thermTime):

        fileCounter = 0

        for i in os.listdir(dataDir):
            if i[0:5] == "data-" and i[-4:] == ".csv":
                fileCounter += 1

        subprocess.check_call(
            [
                "./su2McEvaluator.R",
                dataDir,
                str(thermTime),
                str(fileCounter),
                outputFile,
            ],
            stdout=subprocess.DEVNULL,
        )
