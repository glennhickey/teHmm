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
from teHmm.common import runParallelShellCommands, getLocalTempDir

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
tracksPath="tracks.xml"
tracksPath250="tracks_bin250.xml"
genomePath="alyrata.bed"
regions = [["scaffold_1"], ["scaffold_2"], ["scaffold_3"], ["scaffold_4"],
           ["scaffold_5"], ["scaffold_6"], ["scaffold_7"], ["scaffold_8"],
           ["scaffold_9"]]
cutTrack = "polyN"
truthPaths=["alyrata_hollister_clean_gapped_TE.bed", "alyrata_chaux_clean_gapped_TE.bed", "alyrata_repet_gapped_TE.bed"]
modelerPath=["alyrata_modeler_clean_gapped_TE.bed"]

numParallelBatch = 4
cutTrackLenFilter = 100

# HMM options ############
segOpts = "--cutMultinomial --thresh 2"
segLen = 20
numStates = 40
trainThreads = 6
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
    outFile = os.path.join(outDir, "%s_%s%s" % (inName, regionName, inExt))
    if len(suffix) > 0:
        outFile += "_" + suffix
    return outFile

def cutBedRegion(sequenceList, cutTrackPath, inBed, outBed):
    """ grep out a list of sequences and subtract N's """
    tempPath = getLocalTempDir("Temp_cut", ".bed")
    runShellCommand("rm -f %s" % outBed)
    for sequence in sequenceList:
        runShellCommand("grep %s %s >> %s" % (sequence, inBed, tempPAth))
    runShellCommand("subtractBed -a %s -b %s | sortBed > %s" % (tempPath,
                                                                cutBed, outBed))

def filterCutTrack(trackListPath, cutTrackName, cutTrackLenFilter):
    """ return path of length filtered cut track"""
    tracks = TrackList(tracksListPath)
    track = tracks.getTrackByName(cutTrackName)
    assert track is not None
    cutTrackOriginalPath track.getPath()
    cutTrackPath = getOutPath(cutTrackOriginalPath, outDir,
                           "_filter %d" % cutTrackLenFilter)
    runShellCommand("filterBedLen.py %s %s > %s" % (cutTrackOriginalPath,
                                                    cutTrackLenFilter,
                                                    cutTrackPath))
    return cutTrackPath

def cutInput(genomePath, regions, truthPaths, modelerPath,outDir, cutTrackPath):
    """ cut all the input into outDir """
    inList = [genomePath] + regions + truthPaths + [modelerPath]
    assert len(inList) = 1 + len(regions) + len(truthPaths)
    for i, region in enumerate(regions):
        regionName = "region_%d" % i
        for bedFile in inList:
            outFile = getOutPath(inBed, outDir, regionName)
            cutBedRegion(region, cutTrackPath, bedFile, outFile)

def segmentCommands(genomePath, regions, outDir, segOpts, tracksPath)
    """ make the segmenation command list """
    segmentCmds = []
    for i, region in enumerate(regions):
        regionName = "region_%d" % i
        inBed = getOutPath(genomePath, outDir, regionName)
        outBed = getOutPath(genomePath, outDir, regionName, "segments")
        cmd = "segmentTracks.py %s %s %s %s" % (tracksPath,
                                                inBed, outBed, segOpts)
        segmentCmds.append(cmd)
    return segmentCmds


def trainCommands(genomePath, regions, outDir, tracksPath250, segLen, numStates,
                  trainThreads, thresh, numIter):
    trainCmds = []
    for i, region in enumerate(regions):
        regionName = "region_%d" % i
        segmentsBed = getOutPath(genomePath, outDir, regionName, "segments")
        outMod = getOutPath(genomePath, outDir, regionName, "unsup").replace(
            ".bed", ".mod")
        cmd = "teHmmTrain.py %s %s %s" % (tracksPath250, segmentsBed, outMod)
        cmd += " --fixStart"
        cmd += " --segLEn %d" % segLen
        cmd += " --emStates %d" % numStates
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
        regionName = "region_%d" % i
        segmentsBed = getOutPath(genomePath, outDir, regionName, "segments")
        inMod = getOutPath(genomePath, outDir, regionName, "unsup").replace(
            ".bed", ".mod")
        outEvalBed = getOutPath(genomePath, outDir, regionName, "unsup_eval")
        cmd = "teHmmEval.py %s %s %s --bed %s --segment" % (tracksPath250,
                                                            inMod,
                                                            segmentsBed,
                                                            outEvalBed)
        evalCmds.append(cmd)
    return evalCmds
    
def fitCommands(genomePath, regions, outDir, modelerPath, truthPaths):
    fitCmds = []
    for i, region in enumerate(regions):
        regionName = "region_%d" % i
        evalBed = getOutPath(genomePath, outDir, regionName, "unsup_eval")
        fitTgts = [modelerPath] + truthPaths
        for fitTgt in fitTgts:
            fitInputBed = getOutPath(fitTgt, outDir, regionName)
            fitName = os.path.basename(fitInputBed).splitext()[0].replace(
                regionName, "")
            outFitBed = getOutPath(genomePath, outDir, regionName,
                                   "unsup_eval_fit_%s" % fitName)
            outFitLog = outFitBed.replace(".bed", ".log")
            cmd = "fitStateNames.py %s %s %s %s 2> %s --logDebug" % (
                fitInputBed, evalBed, outFitBed, fitFlags, outFitLog)
            fitCmds.append(cmd)
    return fitCmds

def compareCommands(genomePath, regions, outDir, modelerPath, truthPaths):
    compareCmds = []
    for i, region in enumerate(regions):
        regionName = "region_%d" % i
        fitTgts = [modelerPath] + truthPaths
        for fitTgt in fitTgts:
            truthBed = getOutPath(fitTgt, outDir, regionName)
            fitName = os.path.basename(fitInputBed).splitext()[0].replace(
                regionName, "")
            fitBed = getOutPath(genomePath, outDir, regionName,
                                   "unsup_eval_fit_%s" % fitName)
            compFile = fitBed.replace(".bed", "_comp.txt")
            cmd = "compareBedStates.py %s %s > %s" % (truthBed,
                                                      fitBed,
                                                      compFile)
            compareCmds.append(cmd)
    return compareCmds
                                                        


def harvestSTats(genomePath, regions, outDir, modelerPath, truthPaths):
    pass


cutTrackPath = filterCutTrack(tracksPath, cutTrack, cutTrackLenFilter)


segmentCmds = segmentCommands(genomePath, regions, outDir, segOpts, tracksPath)        

trainCmds = trainCommands(genomePath, regions, outDir, tracksPath250,
                              segLen, numStates, trainThreads, thresh, numIter)

fitCmds = fitCommands(genomePath, regions, outDir, modelerPath, truthPaths)

compareCmds =  compareCommands(genomePath, regions, outDir, modelerPath,
                               truthPaths)


print segmentCommands
#runParallelShellCommands(segmentCmds, batchThreads)
print trainCmds
#runParallelShellCommands(trainCmds, batchThreads)
print fitCmds
#runParallelShellCommands(fitCmds, batchThreads)
print compareCmds
#runParallelShellCommands(compareCmds, batchThreads)

harvestStats(genomePath, regions, outDir, modelerPath,
                               truthPaths)


