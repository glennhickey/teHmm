#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import random
import numpy as np

from teHmm.common import runShellCommand, setLogLevel, addLoggingFileHandler
from teHmm.common import runParallelShellCommands
from teHmm.bin.compareBedStates import extractCompStatsFromFile

#
# updated version of multiBench.py. we forget about semi-supervised and
# unsupervised training, but add some more fitting and comparison options
# as well as more automatic slicing and dicing of input
# also write output to csv.  
#

setLogLevel("INFO")
addLoggingFileHandler("log.txt", False)
logOpts = " --logInfo --logFile log.txt "

startPoint = 1 # Segment
#startPoint = 2 # Train
#startPoint = 3 # Eval
#startPoint = 4 # Fit
#startPoint = 5 # Compare
#startPoint = 6 # Munge Stats

# input ############
segTracksPath = "segTracks.xml"
trainTracksPath = "hmmTracks.xml"
compTracksPath = "hmmTracks.xml"

trainRegionPath = "train_region.bed"
evalRegionPath = "eval_region.bed"

modelerPath="../alyrata_repeatmodeler_clean_gapped_TE.bed"
truthPaths=["../alyrata_chaux_clean_gapped_TE.bed", "../alyrata_hollister_clean_gapped_TE.bed", "../alyrata_repet_gapped_TE.bed"]
truthNames=["chaux", "hollister", "repet"]
assert len(truthPaths) == len(truthNames)

# params ###########
maskK = 10000
segOpts = "--cutMultinomial --thresh 2 --delMask %d %s" % (maskK, logOpts)
segLen = 20
numStates = 35
threads = 5
iter = 200
thresh = 0.08
fitFlags = "--tgt TE --qualThresh 0.15 --tl %s %s" % (trainTracksPath, logOpts)
fitFlagsFdr = "--tgt TE --fdr 0.65 --tl %s %s" % (trainTracksPath, logOpts)
interpolateFlags =  "--tgts TE --maxLen %d %s" % (maskK, logOpts)
compIdx = 0 #base
#compIdx = 1 #interval
#compIdx = 2 #weightintervs

#####################

# segment ##########
trainSegPath = "train_segments.bed"
evalSegPath = "eval_segments.bed"
if startPoint <= 1:
    cmdTrain = "segmentTracks.py %s %s %s %s" % (segTracksPath, trainRegionPath, trainSegPath, segOpts)
    cmdEval = "segmentTracks.py %s %s %s %s" % (segTracksPath, evalRegionPath, evalSegPath, segOpts)
    runParallelShellCommands([cmdEval, cmdTrain], 2)

# train ############
modelPath = "hmm.mod"
if startPoint <=2:
    cmd = "teHmmTrain.py %s %s %s %s" % (trainTracksPath, trainSegPath, modelPath, logOpts)
    cmd += " --fixStart"
    cmd += " --segLen %d" % segLen
    cmd += " --numStates %d" % numStates
    cmd += " --reps %d --numThreads %d" % (threads, threads)
    cmd += " --emThresh %f" % thresh
    cmd += " --iter %d" % iter
    cmd += " --segment %s" % trainSegPath
    runShellCommand(cmd)

# eval ############
evalPath = "eval.bed"
if startPoint <=3:
    cmd = "teHmmEval.py %s %s %s --bed %s --segment %s" % (trainTracksPath, modelPath, evalSegPath, evalPath, logOpts)
    runShellCommand(cmd)
    
# fit ############
fitPath = "fit.bed"
fitFdrPath = "fitFdr.bed"
labelPath = "label.bed"
if startPoint <=4:
    runShellCommand("intersectBed -a %s -b %s | sortBed > %s" % (modelerPath, evalRegionPath, labelPath))
    fitCmd = "fitStateNames.py %s %s %s %s" % (labelPath, evalPath, fitPath, fitFlags)
    fitFdrCmd = "fitStateNames.py %s %s %s %s" % (labelPath, evalPath, fitFdrPath, fitFlagsFdr)
    runParallelShellCommands([fitCmd, fitFdrCmd], 2)
    
# compare ############
compDir = "comp"
if not os.path.exists(compDir):
    runShellCommand("mkdir %s" % compDir)
def getTruthPath(idx):
    return os.path.join(compDir, truthNames[idx] + ".bed")

fitPathMI = "fitMI.bed"
fitFdrPathMI = "FitFdrMI.bed"
predNames = ["modeler", "hmmFit", "hmmFitFdr", "hmmFitMI", "hmmFitFdrMI"]
predPaths = [labelPath, fitPath, fitFdrPath, fitPathMI, fitFdrPathMI]
interpolations = [False, False, False, True, True]
fits = ["NA", "F1", "Rec", "F1", "Rec"]

def getCompPath(truthIdx, predIdx):
    return os.path.join(compDir, predNames[predIdx] + "_vs_" + truthNames[truthIdx] + ".txt")

if startPoint <= 5:
    # cut up truths
    for i, truthInputPath in enumerate(truthPaths):
        runShellCommand("intersectBed -a %s -b %s | sortBed > %s" % (truthInputPath, evalRegionPath, getTruthPath(i)))

    # do our interpolations
    runShellCommand("interpolateMaskedRegions.py %s %s %s %s %s" % (
        compTracksPath, evalRegionPath, fitPath, fitPathMI, interpolateFlags))
    runShellCommand("interpolateMaskedRegions.py %s %s %s %s %s" % (
        compTracksPath, evalRegionPath, fitFdrPath, fitFdrPathMI, interpolateFlags))

    # do our comparisons
    compCmds = []
    for i, predName in enumerate(predNames):
        maskFlags = ""
        if interpolations[i] is True:
            maskFlags = " --delMask %d" % maskK
        for j, truthName in enumerate(truthNames):
            cmd = "compareBedStates.py %s %s --tl %s %s > %s" % (getTruthPath(j), predPaths[i], compTracksPath, maskFlags, getCompPath(j, i))
            compCmds.append(cmd)
    runParallelShellCommands(compCmds, 10)

# munging ############
def prettyAcc(prec, rec):
    f1 = 0.
    if prec + rec > 0:
        f1 = (2. * prec * rec) / (prec + rec)
    return ["%.4f" % prec, "%.4f" % rec, "%.4f" % f1]

if startPoint <= 6:
    statsPath = "stats.csv"
    statsFile = open(statsPath, "w")
    header = ",Fit,Interpolate"
    for truthName in truthNames:
        header += ",%s Prec, %s Rec, %s F1" % (truthName, truthName, truthName)
    statsFile.write(header + "\n")
    
    for i, predName in enumerate(predNames):
        line = "%s, %s, %s, " % (predName, fits[i], interpolations[i])
        for j, truthName in enumerate(truthNames):
            compPath = getCompPath(j, i)
            stats = extractCompStatsFromFile(compPath)[compIdx]
            if "TE" not in stats:
                stats["TE"] = (0,0)
            line += ",".join(prettyAcc(stats["TE"][0], stats["TE"][1]))
        statsFile.write(line + "\n")
    statsFile.close()
