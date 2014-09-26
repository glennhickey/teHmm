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
import string

from teHmm.common import runShellCommand, setLogLevel, addLoggingFileHandler
from teHmm.common import runParallelShellCommands, getLocalTempPath
from teHmm.track import TrackList

# todo: make command-line options?
##################################
##################################

setLogLevel("INFO")
addLoggingFileHandler("log.txt", False)

# results
outDir = "gbOut"
if not os.path.isdir(outDir):
    runShellCommand("mkdir %s" % outDir)

# input data ############
tracksPath="tracks_clean.xml"
tracksPath250="tracks_clean_bin250.xml"
genomePath="alyrata.bed"
regions = [["scaffold_1"], ["scaffold_2"], ["scaffold_3"], ["scaffold_4"],
           ["scaffold_5"], ["scaffold_6"], ["scaffold_7"], ["scaffold_8"],
           ["scaffold_9"]]
#regions = [["scaffold_9"]]
cutTrack = "polyN"
truthPaths=["alyrata_hollister_clean_gapped_TE.bed", "alyrata_chaux_clean_gapped_TE.bed", "alyrata_repet_gapped_TE.bed"]
modelerPath="alyrata_repeatmodeler_clean_gapped_TE.bed"

numParallelBatch = 40
cutTrackLenFilter = 100
fragmentFilterLen = 100000

# HMM options ############
segOpts = "--cutMultinomial --thresh 2"
segLen = 20
numStates = 40
trainThreads = 3
thresh = 0.08
numIter = 200
#mpFlags = "--maxProb --maxProbCut 5"
mpFlags = ""
fitFlags = "--ignoreTgt 0"
#####################

##################################
##################################

def getOutPath(inBed, outDir, regionName, suffix=""):
    """ make file name outDir/inName_inRegion.bed """
    inFile = os.path.basename(inBed)
    inName, inExt = os.path.splitext(inFile)
    outFile = os.path.join(outDir, "%s_%s" % (inName, regionName))
    if len(suffix) > 0:
        outFile += "_" + suffix
    outFile += inExt
    return outFile

def cutBedRegion(sequenceList, cutTrackPath, inBed, outBed):
    """ grep out a list of sequences and subtract N's """
    tempPath = getLocalTempPath("Temp_cut", ".bed")
    runShellCommand("rm -f %s" % outBed)
    for sequence in sequenceList:
        runShellCommand("grep %s %s >> %s" % (sequence, inBed, tempPath))
    runShellCommand("subtractBed -a %s -b %s | sortBed > %s" % (tempPath,
                                                                cutTrackPath, outBed))
    runShellCommand("rm -f %s" % tempPath)

def filterCutTrack(genomePath, fragmentFilterLen, trackListPath, cutTrackName,
                   cutTrackLenFilter):
    """ return path of length filtered cut track"""
    tracks = TrackList(trackListPath)
    track = tracks.getTrackByName(cutTrackName)
    assert track is not None
    cutTrackOriginalPath = track.getPath()
    cutTrackPath = getOutPath(cutTrackOriginalPath, outDir,
                           "filter%d" % cutTrackLenFilter)
    runShellCommand("filterBedLengths.py %s %s > %s" % (cutTrackOriginalPath,
                                                    cutTrackLenFilter,
                                                    cutTrackPath))
    tempPath1 = getLocalTempPath("Temp", ".bed")
    runShellCommand("subtractBed -a %s -b %s | sortBed > %s" % (genomePath,
                                                                cutTrackPath,
                                                                tempPath1))
    tempPath2 = getLocalTempPath("Temp", ".bed")
    S = string.ascii_uppercase + string.digits
    tag = ''.join(random.choice(S) for x in range(200))
    runShellCommand("filterBedLengths.py %s %d --rename %s |grep %s | sortBed> %s" % (
        tempPath1, fragmentFilterLen, tag, tag, tempPath2))
    runShellCommand("cat %s | setBedCol.py 3 N | setBedCol.py 4 0 | setBedCol.py 5 . > %s" % (tempPath2, tempPath1))
    runShellCommand("cat %s | setBedCol.py 3 N | setBedCol.py 4 0 | setBedCol.py 5 . >> %s" % (cutTrackPath, tempPath1))
    runShellCommand("sortBed -i %s > %s" % (tempPath1, tempPath2))
    runShellCommand("mergeBed -i %s > %s" %(tempPath2, cutTrackPath))
    runShellCommand("rm -f %s %s" % (tempPath1, tempPath2))                    
    return cutTrackPath

def cutInput(genomePath, regions, truthPaths, modelerPath,outDir, cutTrackPath):
    """ cut all the input into outDir """
    inList = [genomePath] + truthPaths + [modelerPath]
    assert len(inList) == 2 + len(truthPaths)
    for i, region in enumerate(regions):
        regionName = "region%d" % i
        for bedFile in inList:
            outFile = getOutPath(bedFile, outDir, regionName)
            cutBedRegion(region, cutTrackPath, bedFile, outFile)

def segmentCommands(genomePath, regions, outDir, segOpts, tracksPath):
    """ make the segmenation command list """
    segmentCmds = []
    for i, region in enumerate(regions):
        regionName = "region%d" % i
        inBed = getOutPath(genomePath, outDir, regionName)
        outBed = getOutPath(genomePath, outDir, regionName, "segments")
        cmd = "segmentTracks.py %s %s %s %s --logInfo" % (tracksPath,
                                                inBed, outBed, segOpts)
        segmentCmds.append(cmd)
    return segmentCmds


def trainCommands(genomePath, regions, outDir, tracksPath250, segLen, numStates,
                  trainThreads, thresh, numIter):
    trainCmds = []
    for i, region in enumerate(regions):
        regionName = "region%d" % i
        segmentsBed = getOutPath(genomePath, outDir, regionName, "segments")
        outMod = getOutPath(genomePath, outDir, regionName, "unsup").replace(
            ".bed", ".mod")
        cmd = "teHmmTrain.py %s %s %s" % (tracksPath250, segmentsBed, outMod)
        cmd += " --fixStart"
        cmd += " --segLen %d" % segLen
        cmd += " --numStates %d" % numStates
        cmd += " --reps %d --numThreads %d" % (trainThreads, trainThreads)
        cmd += " --emThresh %f" % thresh
        cmd += " --iter %d" % numIter
        cmd += " --segment %s" % segmentsBed
        cmd += " --logInfo --logFile %s" % (outMod.replace(".bed", ".log"))
        trainCmds.append(cmd)
    return trainCmds

def evalCommands(genomePath, regions, outDir, tracksPath250):
    evalCmds = []
    for i, region in enumerate(regions):
        regionName = "region%d" % i
        segmentsBed = getOutPath(genomePath, outDir, regionName, "segments")
        inMod = getOutPath(genomePath, outDir, regionName, "unsup").replace(
            ".bed", ".mod")
        outEvalBed = getOutPath(genomePath, outDir, regionName, "unsup_eval")
        cmd = "teHmmEval.py %s %s %s --bed %s --segment --logInfo" % (tracksPath250,
                                                            inMod,
                                                            segmentsBed,
                                                            outEvalBed)
        evalCmds.append(cmd)
    return evalCmds
    
def fitCommands(genomePath, regions, outDir, modelerPath, truthPaths):
    fitCmds = []
    for i, region in enumerate(regions):
        regionName = "region%d" % i
        evalBed = getOutPath(genomePath, outDir, regionName, "unsup_eval")
        fitTgts = truthPaths
        fitTgts.append(modelerPath)
        for fitTgt in fitTgts:
            fitInputBed = getOutPath(fitTgt, outDir, regionName)
            fitName = os.path.splitext(os.path.basename(
                fitInputBed))[0].replace(regionName, "")
            outFitBed = getOutPath(genomePath, outDir, regionName,
                                   "unsup_eval_fit_%s" % fitName)
            outFitLog = outFitBed.replace(".bed", ".log")
            cmd = "fitStateNames.py %s %s %s %s --logDebug 2> %s" % (
                fitInputBed, evalBed, outFitBed, fitFlags, outFitLog)
            fitCmds.append(cmd)
    return fitCmds

def compareCommands(genomePath, regions, outDir, modelerPath, truthPaths):
    compareCmds = []
    for i, region in enumerate(regions):
        regionName = "region%d" % i
        fitTgts = truthPaths
        fitTgts.append(modelerPath)
        for fitTgt in fitTgts:
            fitInputBed = getOutPath(fitTgt, outDir, regionName)
            fitName = os.path.splitext(os.path.basename(
                fitInputBed))[0].replace(regionName, "")

            truthBed = getOutPath(fitTgt, outDir, regionName)
            
            fitBed = getOutPath(genomePath, outDir, regionName,
                                   "unsup_eval_fit_%s" % fitName)
            compFile = fitBed.replace(".bed", "comp.txt")
            cmd = "compareBedStates.py %s %s > %s" % (truthBed,
                                                      fitBed,
                                                      compFile)
            compareCmds.append(cmd)
    return compareCmds
                                                        


def harvestSTats(genomePath, regions, outDir, modelerPath, truthPaths):
    pass


cutTrackPath = filterCutTrack(genomePath, fragmentFilterLen, tracksPath,
                              cutTrack, cutTrackLenFilter)
cutInput(genomePath, regions, truthPaths, modelerPath, outDir, cutTrackPath)

segmentCmds = segmentCommands(genomePath, regions, outDir, segOpts, tracksPath)        

trainCmds = trainCommands(genomePath, regions, outDir, tracksPath250,
                              segLen, numStates, trainThreads, thresh, numIter)

evalCmds = evalCommands(genomePath, regions, outDir, tracksPath250)

fitCmds = fitCommands(genomePath, regions, outDir, modelerPath, truthPaths)

compareCmds =  compareCommands(genomePath, regions, outDir, modelerPath,
                               truthPaths)


print segmentCmds
runParallelShellCommands(segmentCmds, numParallelBatch)
print trainCmds
runParallelShellCommands(trainCmds, max(1, numParallelBatch / trainThreads))
print evalCmds
runParallelShellCommands(evalCmds, numParallelBatch)
print fitCmds
runParallelShellCommands(fitCmds, numParallelBatch)
print compareCmds
runParallelShellCommands(compareCmds, numParallelBatch)

#harvestStats(genomePath, regions, outDir, modelerPath, truthPaths)


