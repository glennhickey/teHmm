#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import itertools
import copy
import numpy as np

from teHmm.common import runShellCommand
from teHmm.common import runParallelShellCommands
from teHmm.track import TrackList
from teHmm.trackIO import readBedIntervals
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import getLogLevelString, setLogLevel
from teHmm.common import initBedTool, cleanBedTool
from teHmm.bin.compareBedStates import extractCompStatsFromFile
from teHmm.track import TrackList
from teHmm.bin.trackRanking import extractScore
from teHmm.modelIO import loadModel

""" Thin wrapper of teHmmTrain.py and teHmmEval.py to generate a table of
Number-of-HMM-states VS BIC. 
"""

def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=" Thin wrapper of teHmmTrain.py and teHmmEval.py "
        "to generate a table of Number-of-HMM-states VS BIC. Lower BIC"
        " is better")

    parser.add_argument("tracks", help="tracks xml used for training and eval")
    parser.add_argument("trainingBeds", help="comma-separated list of training regions"
                        " (training region size will be a variable in output table). "
                        "if segmentation is activated, these must also be the "
                        "segmented beds...")
    parser.add_argument("evalBed", help="eval region")
    parser.add_argument("trainOpts", help="all teHmmTrain options in quotes")
    parser.add_argument("evalOpts", help="all teHmmEval options in quotes")
    parser.add_argument("states", help="comma separated-list of numbers of states"
                        " to try")
    parser.add_argument("outDir", help="output directory")
    parser.add_argument("--reps", help="number of replicates", type = int,
                        default=1)
    parser.add_argument("--proc", help="maximum number of processors to use"
                        " in parallel", type = int, default = 1)
    parser.add_argument("--resume", help="try not to rewrite existing files",
                        action="store_true", default=False)
                        

    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()
    

    if not os.path.isdir(args.outDir):
        runShellCommand("mkdir %s" % args.outDir)

    # get the sizes of the trianing beds
    trainingSizes = []
    trainingBeds = []
    for tb in  args.trainingBeds.split(","):
        if len(tb) > 0:
            trainingBeds.append(tb)
    for bed in trainingBeds:
        assert os.path.isfile(bed)
        bedLen = 0
        for interval in readBedIntervals(bed):
            bedLen += interval[2] - interval[1]
        trainingSizes.append(bedLen)

    # make sure --bed not in teHmmEval options and --numStates not in train
    # options
    trainOpts = args.trainOpts.split()
    if "--numStates" in args.trainOpts:
        nsIdx = trainOpts.index("--numStates")
        assert nsIdx < len(trainOpts) - 1
        del trainOpts[nsIdx]
        del trainOpts[nsIdx]
    trainProcs = 1
    if "--numThreads" in args.trainOpts:
        npIdx = trainOpts.index("--numThreads")
        assert npIdx < len(trainOpts) - 1
        trainProcs = int(trainOpts[npIdx + 1])
    segOptIdx = -1
    if "--segment" in args.trainOpts:
        segIdx = trainOpts.index("--segment")
        assert segIdx < len(trainOpts) - 1
        segOptIdx = segIdx + 1
    evalOpts = args.evalOpts.split()
    if "--bed" in args.evalOpts:
        bedIdx = evalOpts.index("--bed")
        assert bedIdx < len(evalOpts) - 1
        del evalOpts[bedIdx]
        del evalOpts[bedIdx]
    if "--bic" in args.evalOpts:
        bicIdx = evalOpts.index("--bic")
        assert bicIdx < len(evalOpts) - 1
        del evalOpts[bicIdx]
        del evalOpts[bicIdx]

    trainCmds = []
    evalCmds = []
    prevSize = -1
    sameSizeCount = 0
    for trainingSize, trainingBed in zip(trainingSizes, trainingBeds):
        # hack to take into account we may have different inputs with same
        # same size, so their corresponding results need unique filenames
        if trainingSize == prevSize:
            sameSizeCount += 1
        else:
            sameSizeCount = 0
        prevSize = trainingSize
        print prevSize, trainingSize, sameSizeCount
        for numStates in args.states.split(","):
            for rep in xrange(args.reps):
                outMod = os.path.join(args.outDir, "hmm_%d.%d.%d.%d.mod" % (
                    trainingSize, sameSizeCount, int(numStates), int(rep)))
                if segOptIdx != -1:
                    trainOpts[segOptIdx] = trainingBed
                trainCmd = "teHmmTrain.py %s %s %s %s --numStates %d" % (
                    args.tracks, trainingBed, outMod, " ".join(trainOpts),
                    int(numStates))
                if not args.resume or not os.path.isfile(outMod) or \
                   os.path.getsize(outMod) < 100:
                    trainCmds.append(trainCmd)

                outBic = outMod.replace(".mod", ".bic")
                outBed = outMod.replace(".mod", "_eval.bed")
                evalCmd = "teHmmEval.py %s %s %s --bed %s --bic %s %s" % (
                    args.tracks, outMod, args.evalBed, outBed, outBic,
                    " ".join(evalOpts))
                if not args.resume or not os.path.isfile(outBic) or \
                   os.path.getsize(outBic) < 2:
                    evalCmds.append(evalCmd)
            
    # run the training            
    runParallelShellCommands(trainCmds, max(1, args.proc / trainProcs))

    # run the eval
    runParallelShellCommands(evalCmds, args.proc)

    # make the table header
    tableFile = open(os.path.join(args.outDir, "bictable.csv"), "w")
    tableFile.write("trainSize, states, meanBic, minBic, maxBic")
    for i in xrange(args.reps):
        tableFile.write(", bic.%d" % i)
    tableFile.write("\n")

    # make the table body
    prevSize = -1
    sameSizeCount = 0
    for (trainingSize,trainingBed) in zip(trainingSizes, trainingBeds):
        # hack to take into account we may have different inputs with same
        # same size, so their corresponding results need unique filenames
        if trainingSize == prevSize:
            sameSizeCount += 1
        else:
            sameSizeCount = 0
        prevSize = trainingSize
        for numStates in args.states.split(","):
            bics = []
            printBics = []
            for rep in xrange(args.reps):
                outMod = os.path.join(args.outDir, "hmm_%d.%d.%d.%d.mod" % (
                    trainingSize, sameSizeCount, int(numStates), int(rep)))
                outBic = outMod.replace(".mod", ".bic")
                try:
                    with open(outBic, "r") as obFile:
                        for line in obFile:
                            bic = float(line.split()[0])
                            break
                    bics.append(bic)
                    printBics.append(bic)
                except:
                    logger.warning("Coudn't find bic %s" % outBic)
                    printBics.append("ERROR")
            # write row
            tableFile.write("%d, %d" % (int(trainingSize), int(numStates)))
            if len(bics) > 0:
                tableFile.write(", %f, %f, %f" % (np.mean(bics), np.min(bics),
                                                  np.max(bics)))
            else:
                tableFile.write(", ERROR, ERROR, ERROR")
            for pb in printBics:
                tableFile.write(", %s" % pb)
            tableFile.write("\n")
    tableFile.close()
            
    cleanBedTool(tempBedToolPath)

if __name__ == "__main__":
    sys.exit(main())
