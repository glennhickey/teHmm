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
import itertools

from teHmm.common import runShellCommand, setLogLevel, addLoggingFileHandler
from teHmm.bin.compareBedStates import extractCompStatsFromFile

workPath = "work"
tracksPath = "tracks_bin250.xml"
copyTrack = "copy"
copyRanges = [("All", 0, sys.maxint), ("Low", 0, 19), ("Middle", 20, 99),
              ("High", 100, sys.maxint)]
compCsvPath = os.path.join(workPath, "comp.csv")
statsCsvPath = os.path.join(workPath, "stats.csv")
outCsvPath = "cross.csv"

setLogLevel("INFO")
addLoggingFileHandler("log.txt", False)

bedFiles=dict()
bedFiles["hollister"] = "alyrata_hollister_clean.bed"
bedFiles["modeler"] = "alyrata_repeatmodeler_clean.bed"
bedFiles["chaux"] = "alyrata_chaux_clean.bed"
bedFiles["hmm"] = "hmm_1_clean_2state.bed"
bedFiles["trf"] = "alyrata_trf_clean.bed"
bedFiles["fgenesh"] = "alyrata_fgenesh_clean.bed"

regionPath = "region1c4.bed"

runShellCommand("rm -rf %s; mkdir %s" % (workPath, workPath))

def bedPath(name, s) :
    return os.path.join(workPath, name + "_%s.bed" % s)

# make working files
# _out : intersection
# _te : TE-state renamed TE and everything else removed
# _gap : _out with gaps added
# _gap_te : _te with gaps added
for name, path in bedFiles.items():
    tPath = bedPath(name, "temp")
    outPath = bedPath(name, "out")
    runShellCommand("intersectBed -a %s -b %s > %s" % (path, regionPath, tPath))
    runShellCommand("setScoreFromTrackIntersection.py %s %s %s %s" % (
        tracksPath, tPath, copyTrack, outPath))

    # make TE/(nothing else) version
    twoPath = bedPath(name, "te")
    runShellCommand("rm2State.sh %s | grep TE > %s" % (outPath, twoPath))

    # make gapped version
    gapPath = bedPath(name, "gap")
    runShellCommand("addBedGaps.py %s %s %s" % (regionPath, outPath, gapPath))

    # make gapped TE version
    twoGapPath = bedPath(name, "gap_te")
    runShellCommand("addBedGaps.py %s %s %s" % (regionPath, twoPath, twoGapPath))

# now we need to do all the deltas and intersections
difFiles = dict()
intFiles = dict()
for name1 in bedFiles.keys():
    for name2 in bedFiles.keys():
        if name1 == name2:
            continue
        bedA, bedB = bedPath(name1, "te"), bedPath(name2, "te")
        difName = "%s_minus_%s" % (name1, name2)
        difPath = bedPath(difName, "te")
        runShellCommand("subtractBed -a %s -b %s > %s" % (bedA, bedB, difPath))        
        # make gapped TE version
        twoGapPath = bedPath(difName, "gap_te")
        runShellCommand("addBedGaps.py %s %s %s" % (
            regionPath, difPath, twoGapPath))
        # add to our file list
        difFiles[difName] = difPath

        # make intersection file
        if name1 < name2:
            intName = "%s_int_%s" % (name1, name2)
            intPath = bedPath(intName, "te")
            runShellCommand("intersectBed -a %s -b %s > %s" % (bedA, bedB,
                                                               intPath)) 
            #make gapped TE version
            twoGapPath = bedPath(intName, "gap_te")
            runShellCommand("addBedGaps.py %s %s %s" % (
                regionPath, intPath, twoGapPath))
            # add to our file list
            intFiles[intName] = intPath
            
# do TE/nonTE comparisions
compFile = open(compCsvPath, "w")
for compRange in copyRanges:
    compName, minScore, maxScore = compRange
    prMap = dict()
    prIntMap = dict()
    for compSet in itertools.combinations(bedFiles.keys(), 2):
        compPath = os.path.join(workPath, "%s_%s_comp.txt" % compSet)
        truthPathUnfiltered = bedPath(compSet[0], "gap_te")
        queryPathUnfiltered = bedPath(compSet[1], "gap_te")
        truthPath=bedPath(compSet[0], "gap_te_%s" % compName)
        queryPath = bedPath(compSet[1], "gap_te_%s" % compName)

        runShellCommand("filterBedScores.py %s --names TE %f %f"
                        " --rename 0 > %s" % (
                            truthPathUnfiltered, minScore, maxScore, truthPath))
        runShellCommand("filterBedScores.py %s --names TE %f %f"
                        " --rename 0 > %s" % (
                            queryPathUnfiltered, minScore, maxScore,queryPath))

        runShellCommand("compareBedStates.py %s %s > %s" % (
            truthPath, queryPath, compPath))
        baseStats, intervalStats, weightedStats = extractCompStatsFromFile(compPath)
        if "TE" not in baseStats:
            logger.warning("No TE elements in %s" % compPath)
            baseStats["TE"] = (-1., -1.)
            intervalStats["TE"] = (-1, -1.)
        assert "TE" in baseStats
        if compSet[0] not in prMap:
            prMap[compSet[0]] = dict()
            prIntMap[compSet[0]] = dict()
        prMap[compSet[0]][compSet[1]] = baseStats["TE"]
        prIntMap[compSet[0]][compSet[1]] = intervalStats["TE"]
        if compSet[1] not in prMap:
            prMap[compSet[1]] = dict()
            prIntMap[compSet[1]] = dict()
        prMap[compSet[1]][compSet[0]] = baseStats["TE"][1], baseStats["TE"][0]
        prIntMap[compSet[1]][compSet[0]] = (intervalStats["TE"][1],
                                            intervalStats["TE"][0])

    # write the TE/nonTE comparisions in CSV table
    for batch in [("Base Stats", prMap), ("Interval Stats", prIntMap)]:
        # header
        compFile.write(batch[0] +"(%d <= score <= %d)\n" % (minScore, maxScore))
        compFile.write("query")
        for truth in bedFiles.keys():
            compFile.write(",%s,," % truth)
        compFile.write("\n")
        #body
        for query in bedFiles.keys():
            compFile.write("%s" % query)
            for truth in bedFiles.keys():
                if query == truth:
                    prec, rec, f1 = 1., 1., 1.
                else:
                    prec, rec = batch[1][truth][query]
                    f1 = 0.
                    if prec + rec > 0.:
                        f1 = (2. * prec * rec) / (prec + rec)
                compFile.write(",%.3f,%.3f,%.3f" % (prec, rec, f1))
            compFile.write("\n")
compFile.close()

# Start the stats
runShellCommand("echo Cross Comparison Stats > %s" % outCsvPath)

# now make a combined bed file to run through bedStats
for batch in [("Input Stats", bedFiles), ("Delta Stats", difFiles),
              ("Intersection Stats", intFiles)]:
    runShellCommand("echo %s >> %s" % (batch[0], outCsvPath))
    teCombinePath = bedPath("combined", "te_gap")
    runShellCommand("rm -f %s" % teCombinePath)

    for name in batch[1].keys():
        twoPath = bedPath(name, "te")
        runShellCommand("cat %s | setBedCol.py 3 TE_%s >> %s" % (
            twoPath, name, teCombinePath))

    # todo (combined breakdown by score)
    
    # run the bedstats
    runShellCommand("bedStats.py %s %s --logHist" % (teCombinePath,
                                                     statsCsvPath))

    runShellCommand("cat %s >> %s" % (statsCsvPath, outCsvPath))
    
runShellCommand("echo Cross Comparison Accuracy >> %s" % outCsvPath)
runShellCommand("cat %s >> %s" % (compCsvPath, outCsvPath))
        

          
                     
            




    
                    



regionFile="region1c4.bed"



            
