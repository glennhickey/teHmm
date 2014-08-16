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
compCsvPath = os.path.join(workPath, "comp.csv")
statsCsvPath = os.path.join(workPath, "stats.csv")
outCsvPath = "cross.csv"

setLogLevel("INFO")
addLoggingFileHandler("log.txt", False)

bedFiles=dict()
bedFiles["holl"] = "alyrata_hollister_clean.bed"
bedFiles["modelr"] = "alyrata_repeatmodeler_clean.bed"
bedFiles["chaux"] = "alyrata_chaux_clean.bed"
bedFiles["hmm"] = "hmm_1_clean_2state.bed"

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

# now we need to do all the deltas
difFiles = dict()
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
            
# do TE/nonTE comparisions
prMap = dict()
for compSet in itertools.combinations(bedFiles.keys(), 2):
    compPath = os.path.join(workPath, "%s_%s_comp.txt" % compSet)
    runShellCommand("compareBedStates.py %s %s > %s" % (
        bedPath(compSet[0], "gap_te"), bedPath(compSet[1], "gap_te"), compPath))
    baseStats, intervalStats, weightedStats = extractCompStatsFromFile(compPath)
    assert "TE" in baseStats
    if compSet[0] not in prMap:
        prMap[compSet[0]] = dict()
    prMap[compSet[0]][compSet[1]] = baseStats["TE"]
    if compSet[1] not in prMap:
        prMap[compSet[1]] = dict()
    prMap[compSet[1]][compSet[0]] = baseStats["TE"][1], baseStats["TE"][0]

# write the TE/nonTE comparisions in CSV table
compFile = open(compCsvPath, "w")
# header
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
            prec, rec = prMap[truth][query]
            f1 = 0.
            if prec + rec > 0.:
                f1 = (2. * prec * rec) / (prec + rec)
        compFile.write(",%.3f,%.3f,%.3f" % (prec, rec, f1))
    compFile.write("\n")
compFile.close()

# now make a combined bed file to run through bedStats
teCombinePath = bedPath("combined", "te_gap")
runShellCommand("rm -f %s" % teCombinePath)

for name in bedFiles.keys() + difFiles.keys():
    twoPath = bedPath(name, "te")
    runShellCommand("cat %s | setBedCol.py 3 TE_%s >> %s" % (
        twoPath, name, teCombinePath))

# todo (combined breakdown by score)
    
# run the bedstats
runShellCommand("bedStats.py %s %s --logHist" % (teCombinePath, statsCsvPath))

runShellCommand("echo Cross Comparison Stats > %s" % outCsvPath)
runShellCommand("cat %s >> %s" % (statsCsvPath, outCsvPath))
runShellCommand("echo Cross Comparison Accuracy >> %s" % outCsvPath)
runShellCommand("cat %s >> %s" % (compCsvPath, outCsvPath))
        

          
                     
            




    
                    



regionFile="region1c4.bed"



            
