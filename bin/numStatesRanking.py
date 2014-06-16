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
from pybedtools import BedTool, Interval
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import getLogLevelString, setLogLevel
from teHmm.bin.compareBedStates import extractCompStatsFromFile
from teHmm.track import TrackList
from teHmm.bin.trackRanking import extractScore
from teHmm.modelIO import loadModel

""" Helper script to do an analysis on how well a model performs
with respect to the number of states.  Based on trackRanking.py
"""

def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Helper script to do an analysis on how well a model "
        "performs with respect to the number of states.")

    parser.add_argument("tracks", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("training", help="BED Training regions"
                        "teHmmTrain.py")
    parser.add_argument("truth", help="BED Truth used for scoring")
    parser.add_argument("states", help="States (in truth) to use for"
                        " average F1 score (comma-separated")
    parser.add_argument("modelSizes", help="comma-separated list of "
                        "integers corresponding to the varios model"
                        " sizes to try",)
    parser.add_argument("outDir", help="Directory to place all results")
    parser.add_argument("--benchOpts", help="Options to pass to "
                        "teHmmBenchmark.py (wrap in double quotes)",
                        default="")
    parser.add_argument("--segOpts", help="Options to pass to "
                        "segmentTracks.py (wrap in double quotes)",
                        default="--comp first --thresh 1 --cutUnscaled")
    parser.add_argument("--numPar", help="Number of replicates to"
                        " perform in parallel.  Note this is compounded"
                        " by --numThreads if the latter is passed"
                        " as an option in --benchOpts", type=int,
                        default=1)
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)

    # make sure no no-no options in benchOpts
    if "--eval" in args.benchOpts or "--truth" in args.benchOpts:
        raise RuntimeError("--eval and --truth cannot be passed through to "
                           "teHmmBenchmark.py as they are generated from "
                           "<training> and <truth> args from this script")
    
    # don't want to keep track of extra logic required for not segmenting
    if "--segment" not in args.benchOpts:
        args.benchOpts += " --segment"
        logger.warning("Adding --segment to teHmmBenchmark.py options")

    # make sure benchopts don't contain anything that would contradict
    # what we're doing (ie series of completely unsupervised trials)
    if "--emStates" in args.benchOpts or\
      "--supervised" in args.benchOpts or\
      "--flatEm" in args.benchOpts or\
      "--init" in args.benchOpts:
      raise RuntimeError("Invalid teHmmBenchmark.py option specified {"
                         "--emStates, --supervised, --flatEm, --init*}")

    try:
        args.modelSizes = [int(x) for x in args.modelSizes.split(",")]
        assert len(args.modelSizes) > 0
    except:
        raise RuntimeError("Problem parsing modelSizes argument")
        
    if not os.path.exists(args.outDir):
        os.makedirs(args.outDir)

    statesRank(args)

def statesRank(args):
    """ Iteratively add best track to a (initially empty) tracklist according
    to some metric"""
    inputTrackList = TrackList(args.tracks)
    rankedTrackList = TrackList()

    benchDir = args.outDir

    trainingPath = args.training
    truthPath = args.truth
    tracksPath = os.path.join(benchDir, "tracks.xml")
    runShellCommand("cp %s %s" % (args.tracks, tracksPath))
    
    segLogPath = os.path.join(benchDir, "segment_cmd.txt")
    segLog = open(segLogPath, "w")
   
    # segment training
    segTrainingPath = os.path.join(benchDir,
                                   os.path.splitext(
                                       os.path.basename(trainingPath))[0]+
                                   "_trainSeg.bed")    
    segmentCmd = "segmentTracks.py %s %s %s %s" % (tracksPath,
                                                   trainingPath,
                                                   segTrainingPath,
                                                   args.segOpts)

    runShellCommand(segmentCmd)
    segLog.write(segmentCmd + "\n")
    
    # segment eval
    segEvalPath = os.path.join(benchDir,
                                os.path.splitext(os.path.basename(truthPath))[0]+
                                "_evalSeg.bed")    
    segmentCmd = "segmentTracks.py %s %s %s %s" % (tracksPath,
                                                   truthPath,
                                                   segEvalPath,
                                                   args.segOpts)

    runShellCommand(segmentCmd)
    segLog.write(segmentCmd + "\n")
    
    segLog.close()

    segPathOpts = " --eval %s --truth %s" % (segEvalPath, truthPath)

    # run teHmmBenchmark.py for each number of states
    benchCmdList = []
    
    for modelSize in args.modelSizes:
        trialDir = os.path.join(benchDir, "states%d" % modelSize)
        statesOpts = " --emStates %d" % modelSize
        benchCmd = "teHmmBenchmark.py %s %s %s %s" % (tracksPath,
                                                      trialDir,
                                                      segTrainingPath,
                                                      args.benchOpts + segPathOpts + statesOpts)
        benchCmdList.append(benchCmd)

    runParallelShellCommands(benchCmdList, args.numPar)

    
    # collect the output into a table
    
    makeStatesTable(args, benchDir, segTrainingPath)
    
def makeStatesTable(args, benchDir, segTrainingPath):
    """ make a table of (modelSize, F1-Score, BIC) """

    tablePath = os.path.join(benchDir, "ranking.txt")
    tableFile = open(tablePath, "w")
    for modelSize in args.modelSizes:
        trialDir = os.path.join(benchDir, "states%d" % modelSize)

        #F1 Score
        score = extractScore(trialDir, segTrainingPath, args)

        #BIC
        bicPath = os.path.join(trialDir,
                             os.path.splitext(
                                 os.path.basename(segTrainingPath))[0]+
                                "_bic.txt")
        bicFile = open(bicPath, "r")
        for line in bicFile:
            bic = float(line.split()[0])
            break
        bicFile.close()
        tableFile.write("%d\t%f\t%f\n" % (modelSize, score, bic))

    tableFile.close()


if __name__ == "__main__":
    sys.exit(main())
