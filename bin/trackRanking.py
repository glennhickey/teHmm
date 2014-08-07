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
from teHmm.modelIO import loadModel
from scipy.stats import linregress
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
    parser.add_argument("--fullSegment", help="Only use segmentation"
                        " based on entire track list for each iteration"
                        " rather than compute segmentation each time (as"
                        " done by default)", action="store_true",
                        default=False)
    parser.add_argument("--bic", help="rank by BIC instead of score "
                        " (both always present in output table though)",
                        action="store_true", default=False)
    parser.add_argument("--base", help="use base-level F1 instead of "
                        "interval-level", default=False, action="store_true")
    parser.add_argument("--naive", help="rank by \"naive\" score",
                         action="store_true", default=False)
    parser.add_argument("--doNaive", help="compute naive stats.  will be "
                        "turned on by default if --naive is used", default=False,
                        action="store_true")
    
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

    if args.bic is True and args.naive is True:
        raise RuntimeError("--bic and --naive are mutually incompatible")
    if args.naive is True:
        args.doNaive = True
        
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

    # compute full segmentation if --fullSegment is True
    if args.fullSegment is True:
        args.fullSegTrainPath = os.path.abspath(os.path.join(args.outDir,
                                                             "fullSegTrain.bed"))
        segmentCmd = "segmentTracks.py %s %s %s %s" % (args.tracks,
                                                       args.training,
                                                       args.fullSegTrainPath,
                                                       args.segOpts)
        runShellCommand(segmentCmd)
        args.fullSegEvalPath = os.path.abspath(os.path.join(args.outDir,
                                                            "fullSegEval.bed"))
        segmentCmd = "segmentTracks.py %s %s %s %s" % (args.tracks,
                                                       args.truth,
                                                       args.fullSegEvalPath,
                                                       args.segOpts)
        runShellCommand(segmentCmd)

    #header
    rankFile = open(os.path.join(args.outDir, "ranking.txt"), "w")
    rankFile.write("It.\tTrack\tF1\tBIC\tNaiveF1\tAccProbSlop\tAccProbR2\n")
    rankFile.close()
    
    # baseline score if we not starting from scratch
    baseIt = 0
    if args.startTracks is not None:
        curTrackList = copy.deepcopy(rankedTrackList)
        score,bic,naive,slope,rsq = runTrial(curTrackList, baseIt, "baseline_test", args)
        rankFile = open(os.path.join(args.outDir, "ranking.txt"), "a")
        rankFile.write("%d\t%s\t%s\t%s\t%s\t%s\t%s\n" % (baseIt, args.startTracks,
                                        score, bic, naive,slope,rsq))
        rankFile.close()
        baseIt += 1
        
    for iteration in xrange(baseIt, baseIt + numTracks):
        bestItScore = -sys.maxint
        bestItBic = sys.maxint
        bestItNaive = -sys.maxint
        bestNextTrack = None
        bestSlope = None
        bestR = None
        for nextTrack in inputTrackList:
            if rankedTrackList.getTrackByName(nextTrack.getName()) is not None:
                continue
            curTrackList = copy.deepcopy(rankedTrackList)
            curTrackList.addTrack(nextTrack)
            score,bic,naive,slope,rsq = runTrial(curTrackList, iteration, nextTrack.getName(),
                                args)
            best = False
            if args.bic is True:
                if bic < bestItBic or (bic == bestItBic and score > bestItScore):
                    best = True
            elif args.naive is True:
                if naive > bestItNaive or (naive == bestItNaive and score > bestItScore):
                    best = True
            elif score > bestItScore or (score == bestItScore and bic < bestItBic):
                    best = True
            if best is True:
                bestItScore, bestItBic, bestItNaive, bestSlope, bestR, bestNextTrack =\
                       score, bic, naive, slope, rsq, nextTrack
            flags = "a"
            if iteration == baseIt:
                flags = "w"      
            trackLogFile = open(os.path.join(args.outDir, nextTrack.getName() +
                                             ".txt"), flags)
            trackLogFile.write("%d\t%f\t%f\t%f\t%f\t%f\n" % (iteration, score, bic, naive,
                                                             slope, rsq))
            trackLogFile.close()
        rankedTrackList.addTrack(copy.deepcopy(bestNextTrack))
        rankedTrackList.saveXML(os.path.join(args.outDir, "iter%d" % iteration,
                                "tracks.xml"))
        
        rankFile = open(os.path.join(args.outDir, "ranking.txt"), flags)
        rankFile.write("%d\t%s\t%s\t%s\t%s\t%s\t%s\n" % (iteration, bestNextTrack.getName(),
                                            bestItScore, bestItBic, bestItNaive,
                                            bestSlope, bestR))
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

    if args.fullSegment is False:
        runShellCommand(segmentCmd)
        segLog.write(segmentCmd + "\n")
    else:
        runShellCommand("ln -f -s %s %s" % (args.fullSegTrainPath, segTrainingPath))

    # segment eval
    segEvalPath = os.path.join(benchDir,
                                os.path.splitext(os.path.basename(truthPath))[0]+
                                "_evalSeg.bed")    
    segmentCmd = "segmentTracks.py %s %s %s %s" % (tracksPath,
                                                   truthPath,
                                                   segEvalPath,
                                                   args.segOpts)
    if trainingPath == truthPath:
        segmentCmd = "ln -f -s %s %s" % (args.segTrainingPath, args.segEvalPath)
    if args.fullSegment is False:
        runShellCommand(segmentCmd)
        segLog.write(segmentCmd + "\n")
    else:
        runShellCommand("ln -f -s %s %s" % (args.fullSegEvalPath, segEvalPath))
    
    segLog.close()

    segPathOpts = " --eval %s --truth %s" % (segEvalPath, truthPath)
    
    benchCmd = "teHmmBenchmark.py %s %s %s %s" % (tracksPath,
                                                  benchDir,
                                                  segTrainingPath,
                                                  args.benchOpts + segPathOpts)
    runShellCommand(benchCmd)

    score = extractScore(benchDir, segTrainingPath, args)
    bic = extractBIC(benchDir, segTrainingPath, args)
    naive = 0
    if args.doNaive is True:
        naive = extractNaive(tracksPath, benchDir, segTrainingPath, args)
    slope, rsq = extractF1ProbSlope(benchDir, segTrainingPath, args)

    # clean up big files?

    return score, bic, naive, slope, rsq

def extractScore(benchDir, benchInputBedPath, args, repSuffix = ""):
    """ Reduce entire benchmark output into a single score value """

    compPath = os.path.join(benchDir,
                             os.path.splitext(
                                 os.path.basename(benchInputBedPath))[0]+
                                "_comp.txt" + repSuffix) 
    baseStats, intStats, weightedStats = extractCompStatsFromFile(compPath)
    stats = intStats
    if args.base is True:
        stats = baseStats
    f1List = []
    for state in args.states.split(","):
        if state not in stats:
            logger.warning("State %s not found in intstats %s. giving 0" % (
                state, str(stats)))
            f1List.append(0)
            continue
        
        prec = stats[state][0]
        rec = stats[state][1]
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

def extractTotalProb(benchDir, benchInputBedPath, args, repSuffix=""):
    """ Get the total log probability from the model """
    modPath = os.path.join(benchDir,
                             os.path.splitext(
                                 os.path.basename(benchInputBedPath))[0]+
                                ".mod" + repSuffix)
    model = loadModel(modPath)
    totalProb = model.getLastLogProb()
    return totalProb

def extractF1ProbSlope(benchDir, benchInputBedPath, args):
    """ Get the slope of the line that fits the graph of total log prob Vs Score.
    (where x-axis = prob, y-axis = score)
    Each point here is a training replicate
    RETURNS :  Slope, R-square
    """

    defRetVal = -1000000, 1000000
    if "--saveAllReps" not in args.benchOpts and "--reps" not in args.benchOpts:
        return defRetVal
    modPath = os.path.join(benchDir,
                             os.path.splitext(
                                 os.path.basename(benchInputBedPath))[0]+
                                ".mod")
    compPath = os.path.join(benchDir,
                             os.path.splitext(
                                 os.path.basename(benchInputBedPath))[0]+
                                "_comp.txt")
    probs = []
    scores = []
    for i in xrange(-1, 1000):
        repSuffix = ""
        if i >= 0:
            repSuffix = ".rep%d" % i
        if os.path.isfile(modPath + repSuffix) and \
          os.path.isfile(compPath + repSuffix):
          score = extractScore(benchDir, benchInputBedPath, args, repSuffix)
          prob = extractTotalProb(benchDir, benchInputBedPath, args, repSuffix)
          probs.append(prob)
          scores.append(score)
    if len(probs) < 2:
        return defRetVal

    # scale down probs to ratios to correct for differences
    # between numbers of tracks (i hope)
    minProb = np.min(probs)
    scaleProbs = [x + minProb for x in probs]
    maxProb = np.max(scaleProbs)
    if maxProb > 0:
        scaleProbs = [x / maxProb for x in scaleProbs]

    # fit to line (since prob is log, we probably want to transform, but for
    # these purposes, a postive slope should be a positive slope...
    slope, intercept, r_value, p_value, std_err = linregress(scaleProbs, scores)
    if np.isnan(slope) or slope is None:
        return defRetVal

    plotPath =  os.path.join(benchDir,
                             os.path.splitext(
                                 os.path.basename(benchInputBedPath))[0]+
                                "_f1VsProbReps.txt")
    plotFile = open(plotPath, "w")
    for p, s in zip(probs, scores):
        plotFile.write("%s\t%s\n" % (p, s))
    plotFile.close()

    return slope, r_value ** 2
    

def extractNaive(tracksPath, benchDir, benchInputBedPath, args):
    """ use naiveTrackCombine.py to get a score instead of teHmmBenchmark.py """

    naiveEvalPath = os.path.join(benchDir,
                             os.path.splitext(
                                 os.path.basename(benchInputBedPath))[0]+
                                "_naiveEval.bed")
    naiveFitPath = os.path.join(benchDir,
                             os.path.splitext(
                                 os.path.basename(benchInputBedPath))[0]+
                                "_naiveEval_Fit.bed")
    naiveCompPath = os.path.join(benchDir,
                             os.path.splitext(
                                 os.path.basename(benchInputBedPath))[0]+
                                "_naive_comp.txt")


    runShellCommand("naiveTrackCombine.py %s %s %s" % (tracksPath, args.truth,
                                                        naiveEvalPath))
    runShellCommand("fitStateNames.py %s %s %s" % (args.truth,
                                                   naiveEvalPath,
                                                   naiveFitPath))
    runShellCommand("compareBedStates.py %s %s > %s" % (args.truth, naiveFitPath,
                                                        naiveCompPath))
    score = extractScore(benchDir,
                        naiveCompPath.replace("_naive_comp.txt", "_naive.bed"),
                        args)
    return score


if __name__ == "__main__":
    sys.exit(main())
