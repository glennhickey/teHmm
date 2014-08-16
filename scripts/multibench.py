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

setLogLevel("INFO")
addLoggingFileHandler("log.txt", False)

# input ############
tracksPath="tracks.xml"
tracksPath250="tracks_bin250.xml"
regionPath="region.bed"
truthPath="hollister_region.bed"
trainPath="modeler_region.bed"

superTrackName="repeat_modeler"
segOpts = "--cutMultinomial --thresh 2"
teStates = ["LINE", "SINE", "LTR", "DNA", "RC", "Unknown"]
trainPath2State = "modeler_2state_region.bed"
segLen = 20
numStates = 35
threads = 6
iter = 200
thresh = 0.08
emPrior = 1.0
mpFlags = "--maxProb --maxProbCut 5"
fitFlags = "--ignoreTgt 0 --qualThresh 0.25"
#####################

superTrackPath="tracks_super.xml"
segPath  = "segments.bed"
sedExp = "\"s/" + "\\|".join(teStates) + "/TE/g\""
runShellCommand("rm2State.sh %s > %s" % (trainPath, trainPath2State))

# make a supervised training track
runShellCommand("grep -v %s %s > %s" % (superTrackName, tracksPath250, superTrackPath))

# make a segments
runShellCommand("segmentTracks.py %s %s %s %s --logInfo --logFile log.txt" % (tracksPath, regionPath, segPath, segOpts))

# do a supervised
runShellCommand("mkdir -p supervised")
runShellCommand("teHmmTrain.py  %s %s supervised/out.mod --segment %s --supervised --segLen %d --logInfo" % (
    superTrackPath, trainPath, segPath, segLen))
runShellCommand("teHmmEval.py %s %s %s --bed %s" % (
    superTrackPath, "supervised/out.mod", segPath, "supervised/eval.bed"))
runShellCommand("rm2State.sh %s > %s" % ("supervised/eval.bed", "supervised/evalTE.bed"))
runShellCommand("compareBedStates.py %s %s > %s" % (truthPath, "supervised/evalTE.bed", "supervised/comp.txt"))
runShellCommand("fitStateNames.py %s %s %s %s" % (truthPath, "supervised/eval.bed", "supervised/fit.bed", fitFlags))
runShellCommand("compareBedStates.py %s %s > %s" % (truthPath, "supervised/fit.bed", "supervised/comp_cheat.txt"))
                
# do a semisupervised
runShellCommand("mkdir -p semi")
runShellCommand("createStartingModel.py %s %s %s %s %s --numTot %d --mode full --em %f --outName Unlabeled" % (
    tracksPath, superTrackName, regionPath, "semi/tran.txt", "semi/em.txt", numStates, emPrior))
runShellCommand("grep -v Unlabeled semi/tran.txt > semi/tranf.txt")
runShellCommand("teHmmBenchmark.py %s %s %s --truth %s --iter %d %s --transMatEpsilons --segment --segLen %d --fit --reps %d --numThreads %d --logInfo --fixStart --initTransProbs %s --forceTransProbs %s --initEmProbs %s --forceEmProbs %s --fitOpts \"%s\" " % (
    tracksPath250, "semi/bench", segPath, truthPath, iter, mpFlags, segLen, threads, threads, "semi/tran.txt", "semi/tranf.txt", "semi/em.txt", "semi/em.txt", fitFlags))
evalPath = "semi/bench/" + segPath[:-4] + "_eval.bed"
compPath = "semi/bench/" + segPath[:-4] + "_comp.txt"
runShellCommand("rm2State.sh %s > %s" % (evalPath, "semi/eval1.bed"))
runShellCommand("fitStateNames.py %s %s %s %s" % (trainPath2State, "semi/eval1.bed", "semi/fit1.bed", fitFlags))
runShellCommand("compareBedStates.py %s %s > %s" % (truthPath, "semi/fit1.bed", "semi/comp.txt"))
runShellCommand("cp %s %s" % (compPath, "semi/comp_cheat.txt"))

# do a unsupervised
runShellCommand("mkdir -p unsup")
runShellCommand("teHmmBenchmark.py %s %s %s --truth %s --iter %d %s --maxProb --maxProbCut 5 --segment --segLen %s --fit --reps %d --numThreads %d --logInfo --fixStart --emStates %s --fitOpts \"%s\"" % (
    tracksPath250, "unsup/bench", segPath, truthPath, iter, mpFlags, segLen, threads, threads, numStates, fitFlags))
evalPath = "unsup/bench/" + segPath[:-4] + "_eval.bed"
compPath = "unsup/bench/" + segPath[:-4] + "_comp.txt"
runShellCommand("rm2State.sh %s > %s" % (evalPath, "unsup/eval1.bed"))
runShellCommand("fitStateNames.py %s %s %s %s" % (trainPath2State, "unsup/eval1.bed", "unsup/fit1.bed", fitFlags))
runShellCommand("compareBedStates.py %s %s > %s" % (truthPath, "unsup/fit1.bed", "unsup/comp.txt"))
runShellCommand("cp %s %s" % (compPath, "unsup/comp_cheat.txt"))
