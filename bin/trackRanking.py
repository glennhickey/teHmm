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

""" Helper script to rank a list of tracks based on how well they improve
some measure of HMM accuracy, by wrapping teHmmBenchmark.py
"""

def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Helper script to rank a list of tracks based on how well "
        "they improve some measure of HMM accuracy, by wrapping "
         "teHmmBenchmark.py")

    parser.add_argument("tracks", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("training", help="BED Training regions"
                        "teHmmTrain.py")
    parser.add_argument("truth", help="BED Truth used for scoring")
    parser.add_argument("states", help="States (in truth) to use for"
                        " average F1 score (comma-separated")
    parser.add_argument("outDir", help="Directory to place all results")
    parser.add_argument("--benchOpts", help="Options to pass to "
                        "teHmmBenchmark.py (wrap in double quotes)",
                        default="")
    parser.add_argument("--startTracks", help="comma-separated list of "
                        "tracks to start off with", default = None)
    parser.add_argument("--segOpts", help="Options to pass to "
                        "segmentTracks.py (wrap in double quotes)",
                        default="--comp first --thresh 1 --cutUnscaled")
    parser.add_argument("--bic", help="rank by BIC instead of score "
                        " (both always present in output table though)",
                        action="store_true", default=False)
    
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
        
    if not os.path.exists(args.outDir):
        os.makedirs(args.outDir)

    greedyRank(args)

def greedyRank(args):
    """ Iteratively add best track to a (initially empty) tracklist according
    to some metric"""
    inputTrackList = TrackList(args.tracks)
    rankedTrackList = TrackList()
    if args.startTracks is not None:
        for startTrack in args.startTracks.split(","):
            track = inputTrackList.getTrackByName(startTrack)
            if track is None:
                logger.warning("Start track %s not found in tracks XML" %
                               startTrack)
            else:
                rankedTrackList.addTrack(copy.deepcopy(track))
            
    numTracks = len(inputTrackList) - len(rankedTrackList)
    currentScore, currentBIC = 0.0, sys.maxint

    # baseline score if we not starting from scratch
    baseIt = 0
    if args.startTracks is not None:
        curTrackList = copy.deepcopy(rankedTrackList)
        score,bic = runTrial(curTrackList, baseIt, "baseline_test", args)
        rankFile = open(os.path.join(args.outDir, "ranking.txt"), "w")
        rankFile.write("%d\t%s\t%s\n" % (baseIt, args.startTracks,
                                        score))
        rankFile.close()
        baseIt += 1
        
    for iteration in xrange(baseIt, baseIt + numTracks):
        bestItScore = -sys.maxint
        bestItBic = sys.maxint
        bestNextTrack = None
        for nextTrack in inputTrackList:
            if rankedTrackList.getTrackByName(nextTrack.getName()) is not None:
                continue
            curTrackList = copy.deepcopy(rankedTrackList)
            curTrackList.addTrack(nextTrack)
            score,bic = runTrial(curTrackList, iteration, nextTrack.getName(),
                                args)
            if args.bic is False:
                if score > bestItScore or (score == bestItScore and bic < bestItBic):
                    bestItScore, bestItBic, bestNextTrack = score, bic, nextTrack
            else:
                if bic < bestItBic or (bic == bestItBic and score > bestItScore):
                    bestItScore, bestItBic, bestNextTrack = score, bic, nextTrack
                    
            flags = "a"
            if iteration == 0:
                flags = "w"
            trackLogFile = open(os.path.join(args.outDir, nextTrack.getName() +
                                             ".txt"), flags)
            trackLogFile.write("%d\t%f\t%f\n" % (iteration, score, bic))
            trackLogFile.close()
        rankedTrackList.addTrack(copy.deepcopy(bestNextTrack))
        rankedTrackList.saveXML(os.path.join(args.outDir, "iter%d" % iteration,
                                "tracks.xml"))
        
        rankFile = open(os.path.join(args.outDir, "ranking.txt"), flags)
        rankFile.write("%d\t%s\t%s\t%s\n" % (iteration, bestNextTrack.getName(),
                                            bestItScore, bestItBic))
        rankFile.close()


def runTrial(tracksList, iteration, newTrackName, args):
    """ compute a score for a given set of tracks using teHmmBenchmark.py """
    benchDir = os.path.join(args.outDir, "iter%d" % iteration)
    benchDir = os.path.join(benchDir, "%s_bench" % newTrackName)
    if not os.path.exists(benchDir):
        os.makedirs(benchDir)

    trainingPath = args.training
    truthPath = args.truth

    tracksPath =  os.path.join(benchDir, "tracks.xml")
    tracksList.saveXML(tracksPath)

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
    
    benchCmd = "teHmmBenchmark.py %s %s %s %s" % (tracksPath,
                                                  benchDir,
                                                  segTrainingPath,
                                                  args.benchOpts + segPathOpts)
    runShellCommand(benchCmd)

    score = extractScore(benchDir, segTrainingPath, args)
    bic = extractBIC(benchDir, segTrainingPath, args)

    # clean up big files?

    return score, bic

def extractScore(benchDir, benchInputBedPath, args):
    """ Reduce entire benchmark output into a single score value """

    compPath = os.path.join(benchDir,
                             os.path.splitext(
                                 os.path.basename(benchInputBedPath))[0]+
                                "_comp.txt") 
    baseStats, intStats, weightedStats = extractCompStatsFromFile(compPath)
    f1List = []
    for state in args.states.split(","):
        if state not in intStats:
            logger.warning("State %s not found in intstats %s. giving 0" % (
                state, str(intStats)))
            f1List.append(0)
            continue
        
        prec = intStats[state][0]
        rec = intStats[state][1]
        f1 = 0
        if prec + rec > 0:
            f1 = 2. * ((prec * rec) / (prec + rec))
        f1List.append(f1)

    avgF1 = np.mean(f1List)
    return avgF1

def extractBIC(benchDir, benchInputBedPath, args):
    """ Get the BIC score fro mthe teHmmBenchmark output mess """
        
    bicPath = os.path.join(benchDir,
                             os.path.splitext(
                                 os.path.basename(benchInputBedPath))[0]+
                                "_bic.txt")
    bicFile = open(bicPath, "r")
    for line in bicFile:
        bic = float(line.split()[0])
        break
    bicFile.close()
    return bic


if __name__ == "__main__":
    sys.exit(main())
