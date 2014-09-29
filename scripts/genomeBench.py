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
import copy

from teHmm.common import runShellCommand, setLogLevel, addLoggingFileHandler
from teHmm.common import runParallelShellCommands, getLocalTempPath
from teHmm.track import TrackList
from teHmm.trackIO import bedRead
from teHmm.bin.compareBedStates import extractCompStatsFromFile

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
#regions = [["scaffold_1"], ["scaffold_2"]]
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
fitFlags = "--ignoreTgt 0 --qualThresh 0.2"
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
        fitTgts = copy.deepcopy(truthPaths)
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
        modelerInputBed = getOutPath(modelerPath, outDir, regionName)
        modelerName = os.path.splitext(os.path.basename(
            modelerInputBed))[0].replace(regionName, "")
        modelerFitBed = getOutPath(genomePath, outDir, regionName,
                                   "unsup_eval_fit_%s" % modelerName)
        for fitTgt in fitTgts:
            fitInputBed = getOutPath(fitTgt, outDir, regionName)
            fitName = os.path.splitext(os.path.basename(
                fitInputBed))[0].replace(regionName, "")

            truthBed = getOutPath(fitTgt, outDir, regionName)

            # "Cheat" comparision where we fit with truth
            fitBed = getOutPath(genomePath, outDir, regionName,
                                   "unsup_eval_fit_%s" % fitName)
            compFile = fitBed.replace(".bed", "comp_cheat.txt")
            cmd = "compareBedStates.py %s %s > %s" % (truthBed,
                                                      fitBed,
                                                      compFile)
            compareCmds.append(cmd)
            
            # "Semisupervised" comparison where we fit with modeler
            compFile = fitBed.replace(".bed", "comp_modfit.txt")
            cmd = "compareBedStates.py %s %s > %s " % (truthBed,
                                                       modelerFitBed,
                                                       compFile)
            compareCmds.append(cmd)

            # Modeler comparision with truth (baseline)
            compFile = fitBed.replace(".bed", "_comp_baseline.txt")
            cmd = "compareBedStates.py %s %s > %s " % (truthBed,
                                                       modelerInputBed,
                                                       compFile)
            compareCmds.append(cmd)
            
    return compareCmds
                                                        
def countBases(bedPath):
    numBases = 0
    for interval in bedRead(bedPath):
        numBases += int(interval[2]) - int(interval[1])
    return numBases

def prettyAcc(prec, rec):
    f1 = 0.
    if prec + rec > 0:
        f1 = (2. * prec * rec) / (prec + rec)
    return ["%.4f" % prec, "%.4f" % rec, "%.4f" % f1]

def harvestStats(genomePath, regions, outDir, modelerPath, truthPaths,
                 outStatsName, compIdx):
    statFile = open(os.path.join(outDir, outStatsName) + ".csv", "w")
    rows = []
    header = []
    totalBases = 0
    for i, region in enumerate(regions):
        if len(rows) == 0:
            header.append("region")
        row = []
        regionName = "region%d" % i
        row.append(regionName)
        fitTgts = truthPaths
        modelerInputBed = getOutPath(modelerPath, outDir, regionName)
        modelerName = os.path.splitext(os.path.basename(
            modelerInputBed))[0].replace(regionName, "")
        modelerFitBed = getOutPath(genomePath, outDir, regionName,
                                   "unsup_eval_fit_%s" % modelerName)
        if len(rows) == 0:
            header.append("numBases")
        numBases = countBases(modelerInputBed)
        totalBases += numBases
        row.append(numBases)

        for fitTgt in fitTgts:
            fitInputBed = getOutPath(fitTgt, outDir, regionName)
            fitName = os.path.splitext(os.path.basename(
                fitInputBed))[0].replace(regionName, "")
            # "Cheat" comparision where we fit with truth
            fitBed = getOutPath(genomePath, outDir, regionName,
                                   "unsup_eval_fit_%s" % fitName)
            cheatCompFile = fitBed.replace(".bed", "comp_cheat.txt")
            # "Semisupervised" comparison where we fit with modeler
            compFile = fitBed.replace(".bed", "comp_modfit.txt")
            # Modeler comparision with truth (baseline)
            modCompFile = fitBed.replace(".bed", "_comp_baseline.txt")
            
            stats = extractCompStatsFromFile(compFile)[compIdx]
            statsCheat = extractCompStatsFromFile(cheatCompFile)[compIdx]
            statsMod =  extractCompStatsFromFile(modCompFile)[compIdx]
            if "TE" not in stats:
                stats["TE"] = (0,0)
                statsCheat["TE"] = (0,0)
                statsMod["TE"] = (0,0)
            
            if len(rows) == 0:
                header += [fitName[:12] + "_ModPrec", fitName[:12]+ "_ModRec",
                           fitName[:12]+ "_ModF1"]
            row += prettyAcc(statsMod["TE"][0], statsMod["TE"][1])
            
            if len(rows) == 0:
                header += [fitName[:12] + "_Prec", fitName[:12]+ "_Rec",
                           fitName[:12]+ "_F1"]
            row += prettyAcc(stats["TE"][0], stats["TE"][1])

            if len(rows) == 0:
                header += [fitName[:12] + "_PrecCheat", fitName[:12]+ "_RecCheat",
                           fitName[:12]+ "_F1Cheat"]
            row += prettyAcc(statsCheat["TE"][0], statsCheat["TE"][1])
            
        if len(rows) == 0:
            rows.append(header)
        rows.append(row)
    # compute weighted averages
    row = ["total", totalBases]
    for col in xrange(2,len(rows[0])):
        val = sum( (float(rows[i][1]) / float(totalBases)) * float(rows[i][col])
                   for i in xrange(1, len(rows)))
        row.append("%.4f" % val)
    rows.append(row)
    
    for row in rows:
        statFile.write(",".join(str(x) for x in row) + "\n")
    statFile.close()


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
print "\n".join(compareCmds)
runParallelShellCommands(compareCmds, numParallelBatch)

harvestStats(genomePath, regions, outDir, modelerPath, truthPaths, "stats_base", 0)
harvestStats(genomePath, regions, outDir, modelerPath, truthPaths, "stats_interval", 1)
harvestStats(genomePath, regions, outDir, modelerPath, truthPaths, "stats_weighted", 2)


