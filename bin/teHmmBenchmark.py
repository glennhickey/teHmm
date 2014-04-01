#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import itertools
import copy

from teHmm.common import runShellCommand
from teHmm.common import runParallelShellCommands
from teHmm.track import TrackList
from pybedtools import BedTool, Interval
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import getLogLevelString

""" This script automates evaluating the hmm te model by doing training,
parsing, comparing back to truth, and summerizing the resutls in a table all
in one.  It can run the same logic on multiple input beds at once in parallel
(by using, say, a wildcard argument for inBeds. It also optionally repeats the
evaluation for subsets of the input tracks.

Independent processes are run in parallel using Python's process pool with the
maximum number of parallel processes limited by the --numProc argument
"""
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train, evalaute, then compare hmm model on input")

    parser.add_argument("trainingTracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks used "
                        "for training")
    parser.add_argument("evalTracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks used"
                        " for evaluation")
    parser.add_argument("outputDir", help="directory to write output")
    parser.add_argument("inBeds", nargs="*", help="list of training beds")
    parser.add_argument("--numProc", help="Max number of processors to use",
                        type=int, default=1)
    parser.add_argument("--allTrackCombinations", help="Rerun with all"
                        " possible combinations of tracks from the input"
                        " tracksInfo file.  Note that this number gets big"
                        " pretty fast.", action = "store_true", default= False)
    parser.add_argument("--emStates", help="By default the supervised mode"
                        " of teHmmTrain is activated.  This option overrides"
                        " that and uses the EM mode and the given number of "
                        "states instead", type=int, default=None)
    parser.add_argument("--cross", help="Do 50/50 cross validation by training"
                        " on first half input and validating on second",
                        action="store_true", default=False)
    parser.add_argument("--emFac", help="Normalization factor for weighting"
                        " emission probabilities because when there are "
                        "many tracks, the transition probabilities can get "
                        "totally lost. 0 = no normalization. 1 ="
                        " divide by number of tracks.  k = divide by number "
                        "of tracks / k", type=int, default=0)
    parser.add_argument("--mod", help="Path to trained model.  This will "
                        "bypass the training phase that would normally be done"
                        " and just skip to the evaluation.  Note that the user"
                        " must make sure that the trained model has the "
                        "states required to process the input data",
                        default = None)
    parser.add_argument("--iter", help="Number of EM iterations.  Needs to be"
                        " used in conjunction with --emStates to specify EM"
                        " training",
                        type = int, default=None)
    parser.add_argument("--initTransProbs", help="Path of text file where each "
                        "line has three entries: FromState ToState Probability"
                        ".  This file (all other transitions get probability 0)"
                        " is used to specifiy the initial transition model."
                        " The names and number of states will be initialized "
                        "according to this file (overriding --numStates)",
                        default = None)
    parser.add_argument("--fixTrans", help="Do not learn transition parameters"
                        " (best used with --initTransProbs)",
                        action="store_true", default=False)
    parser.add_argument("--initEmProbs", help="Path of text file where each "
                        "line has four entries: State Track Symbol Probability"
                        ".  This file (all other emissions get probability 0)"
                        " is used to specifiy the initial emission model. All "
                        "states specified in this file must appear in the file"
                        " specified with --initTransProbs (but not vice versa).",
                        default = None)
    parser.add_argument("--fixEm", help="Do not learn emission parameters"
                        " (best used with --initEmProbs)",
                        action="store_true", default=False)
    parser.add_argument("--initStartProbs", help="Path of text file where each "
                        "line has two entries: State Probability"
                        ".  This file (all other start probs get probability 0)"
                        " is used to specifiy the initial start dist. All "
                        "states specified in this file must appear in the file"
                        " specified with --initTransProbs (but not vice versa).",
                        default = None)
    parser.add_argument("--fixStart", help="Do not learn emission parameters"
                        " (best used with --initStartProbs)",
                        action="store_true", default=False)
    parser.add_argument("--forceTransProbs",
                        help="Path of text file where each "
                        "line has three entries: FromState ToState Probability" 
                        ". These transition probabilities will override any "
                        " learned probabilities after training (unspecified "
                        "will not be set to 0 in this case. the learned values"
                        " will be kept, but normalized as needed" ,
                        default=None)
    parser.add_argument("--forceEmProbs", help="Path of text file where each "
                        "line has four entries: State Track Symbol Probability"
                        ". These "
                        "emission probabilities will override any learned"
                        " probabilities after training (unspecified "
                        "will not be set to 0 in this case. the learned values"
                        " will be kept, but normalized as needed." ,
                        default = None) 
    parser.add_argument("--flatEm", help="Use a flat emission distribution as "
                        "a baseline.  If not specified, the initial emission "
                        "distribution will be randomized by default.  Emission"
                        " probabilities specified with --initEmpProbs or "
                        "--forceEmProbs will never be affected by randomizaiton"
                        ".  The randomization is important for Baum Welch "
                        "training, since if two states dont have at least one"
                        " different emission or transition probability to begin"
                        " with, they will never learn to be different.",
                        action="store_true", default=False)
    parser.add_argument("--mandTracks", help="Mandatory track names for use "
                        "with --allTrackCombinations in comma-separated list",
                        default=None)
    parser.add_argument("--combinationRange", help="in form MIN,MAX: Only "
                        "explore track combination in given (closed) range. "
                        "A more refined version of --allTrackCombinations.",
                        default=None)
    parser.add_argument("--supervised", help="Use name (4th) column of "
                        "<traingingBed> for the true hidden states of the"
                        " model.  Transition parameters will be estimated"
                        " directly from this information rather than EM."
                        " NOTE: The number of states will be determined "
                        "from the bed.",
                        action = "store_true", default = False)
    parser.add_argument("--segment", help="Bed file of segments to treat as "
                        "single columns for HMM (ie as created with "
                        "segmentTracks.py).  IMPORTANT: this file must cover "
                        "the same regions as the traininBed file. Unless in "
                        "supervised mode, probably best to use same bed file "
                        " as both traingBed and --segment argument.  Otherwise"
                        " use intersectBed to make sure the overlap is exact",
                        default=None)
        
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    logOps = "--logLevel %s" % getLogLevelString()
    if args.logFile is not None:
        logOps += " --logFile %s" % args.logFile

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)

    trainingTrackList = TrackList(args.trainingTracksInfo)
    evalTrackList = TrackList(args.evalTracksInfo)
    checkTrackListCompatible(trainingTrackList, evalTrackList)

    sizeRange = (len(trainingTrackList), len(trainingTrackList) + 1)
    if args.allTrackCombinations is True:
        sizeRange = (1, len(trainingTrackList) + 1)
    if args.combinationRange is not None:
        toks = args.combinationRange.split(",")
        sizeRange = int(toks[0]),int(toks[1]) + 1
        logger.debug("manual range (%d, %d) " % sizeRange)
    mandTracks = set()
    if args.mandTracks is not None:
        mandTracks = set(args.mandTracks.split(","))
        logger.debug("mandatory set %s" % str(mandTracks))
    trainFlags = ""
    if args.emStates is not None:
        trainFlags += " --numStates %d" % args.emStates
    if args.supervised is True:
        trainFlags += " --supervised"
        if args.segment is True:
            raise RuntimeError("--supervised not currently compatible with "
                               "--segment")
    trainFlags += " --emFac %d" % args.emFac
    if args.forceEmProbs is not None:
        trainFlags += " --forceEmProbs %s" % args.forceEmProbs
    if args.iter is not None:
        assert args.emStates is not None or args.initTransProbs is not None
        trainFlags += " --iter %d" % args.iter
    if args.initTransProbs is not None:
        trainFlags += " --initTransProbs %s" % args.initTransProbs
    if args.initEmProbs is not None:
        trainFlags += " --initEmProbs %s" % args.initEmProbs
    if args.fixEm is True:
        trainFlags += " --fixEm"
    if args.initStartProbs is not None:
        trainFlags += " --initStartProbs %s" % args.initStartProbs
    if args.fixStart is True:
        trainFlags += " --fixStart"
    if args.forceTransProbs is not None:
        trainFlags += " --forceTransProbs %s" % args.forceTransProbs
    if args.forceEmProbs is not None:
        trainFlags += " --forceEmProbs %s" % args.forceEmProbs
    if args.flatEm is True:
        trainFlags += " --flatEm"

    # write out command line for posteriorty's sake
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
    cmdPath = os.path.join(args.outputDir, "teHmmBenchmark_cmd.txt")
    cmdFile = open(cmdPath, "w")
    cmdFile.write(" ".join(argv) + "\n")
    cmdFile.close()
                           
    #todo: try to get timing for each command
    commands = []
    rows = dict()
    for pn, pList in enumerate(subsetTrackList(trainingTrackList, sizeRange,
                                               mandTracks)):
        if len(pList) == len(trainingTrackList):
            outDir = args.outputDir
        else:
            outDir = os.path.join(args.outputDir, "perm%d" % pn)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        trainingTrackPath = os.path.join(outDir, "training_tracks.xml")
        evalTrackPath = os.path.join(outDir, "eval_tracks.xml")
        pList.saveXML(trainingTrackPath)
        epList = TrackList()
        for track in pList:
            t = copy.deepcopy(evalTrackList.getTrackByName(track.getName()))
            epList.addTrack(t)
        epList.saveXML(evalTrackPath)
        
        for inBed in args.inBeds:
            
            base = os.path.basename(inBed)
            truthBed = inBed
            testBed = inBed
            if args.cross is True:
                truthBed = os.path.join(outDir,
                                        os.path.splitext(base)[0] +
                                        "_truth_temp.bed")
                testBed = os.path.join(outDir,
                                       os.path.splitext(base)[0] +
                                       "_test_temp.bed")
                splitBed(inBed, truthBed, testBed)

                                        
            
            # train
            if args.mod is not None:
                modPath = args.mod
                command = "ls %s" % modPath
            else:
                modPath = os.path.join(outDir,
                                       os.path.splitext(base)[0] + ".mod")
                command = "teHmmTrain.py %s %s %s %s %s" % (trainingTrackPath,
                                                            truthBed,
                                                            modPath,
                                                            logOps,
                                                            trainFlags)
                if args.segment is not None:
                    command += " --segment %s" % truthBed

            # view
            viewPath = os.path.join(outDir,
                                   os.path.splitext(base)[0] + "_view.txt")
            command += " && teHmmView.py %s > %s" % (modPath, viewPath)

            # evaluate
            evalBed = os.path.join(outDir,
                                   os.path.splitext(base)[0] + "_eval.bed")
            command += " && teHmmEval.py %s %s %s --bed %s %s" % (evalTrackPath,
                                                                  modPath,
                                                                  testBed,
                                                                  evalBed,
                                                                  logOps)
            if args.segment is not None:
                command += " --segment"
                
            # compare
            compPath = os.path.join(outDir,
                                    os.path.splitext(base)[0] + "_comp.txt")
            command += " && compareBedStates.py %s %s > %s" % (testBed,
                                                               evalBed,
                                                               compPath)


            # make table row
            rowPath = os.path.join(outDir,
                                   os.path.splitext(base)[0] + "_row.txt")
            if inBed in rows:
                rows[inBed].append(rowPath)
            else:
                rows[inBed] = [rowPath]
            command += " && scrapeBenchmarkRow.py %s %s %s %s %s" % (
                args.trainingTracksInfo,
                trainingTrackPath,
                evalBed,
                compPath,
                rowPath)

            # remember command
            inCmdPath = os.path.join(outDir,
                                    os.path.splitext(base)[0] + "_cmd.txt")
            inCmdFile = open(inCmdPath, "w")
            inCmdFile.write(command + "\n")
            inCmdFile.close()

            commands.append(command)
            
    runParallelShellCommands(commands, args.numProc)
    writeTables(args.outputDir, rows)


def subsetTrackList(trackList, sizeRange, mandTracks):
    """ generate tracklists of all combinations of tracks in the input list
    optionally using size range to limit the different sizes tried. so, for
    example, given input list [t1, t2, t3] and sizeRange=None this
    will gneerate [t1] [t2] [t3] [t1,t2] [t1,t3] [t2,t3] [t1,t2,t3] """
    assert sizeRange[0] > 0
    sizeRange  = (sizeRange[0], min(sizeRange[1], len(trackList) + 1))
    for outLen in xrange(*sizeRange):
        for perm in itertools.combinations([x for x in xrange(len(trackList))],
                                            outLen):
            permList = TrackList()
            mandFound = 0
            for trackNo in perm:
                track = copy.deepcopy(trackList.getTrackByNumber(trackNo))
                permList.addTrack(track)
                if track.getName() in mandTracks:
                    mandFound += 1

            if mandFound == len(mandTracks):
                yield permList

def splitBed(inBed, outBed1, outBed2):
    """ Used for cross validation option.  The first half in input bed gets
    written to outBed1 and the second half to outBed2"""
    inFile = open(inBed, "r")
    numLines = len([x for x in inFile])
    inFile.close()
    inFile = open(inBed, "r")
    cutLine = numLines / 2
    outFile1 = open(outBed1, "w")
    outFile2 = open(outBed2, "w")
    for lineNo, line in enumerate(inFile):
        if numLines == 1 or lineNo < cutLine:
            outFile1.write(line)
        if numLines == 1 or lineNo >= cutLine:
            outFile2.write(line)
    inFile.close()
    outFile1.close()
    outFile2.close()

def checkTrackListCompatible(trainingTrackList, evalTrackList):
    """ Now that we allow a different trackList to be used for training and
    eval, we need to check to make sure that everything's the same but the
    paths"""
    for track1, track2 in zip(trainingTrackList, evalTrackList):
        assert track1.getName() == track2.getName()
        assert track1.getNumber() == track2.getNumber()
        assert track1.getScale() == track2.getScale()
        assert track1.getLogScale() == track2.getLogScale()
        assert track1.getDist() == track2.getDist()

def writeTables(outDir, rows):
    """ Write CSV table for each input bed that was scraped from up from the
    output using scrapeBenchmarkRow.py """
    for inBed, rowPaths in rows.items():
        name = os.path.splitext(os.path.basename(inBed))[0]
        tablePath = os.path.join(outDir, name + "_table.csv")
        tableFile = open(tablePath, "w")
        for i, rowPath in enumerate(rowPaths):
            rowFile = open(rowPath, "r")        
            rowLines = [line for line in rowFile]
            rowFile.close()
            if i == 0:
                tableFile.write(rowLines[0])
            tableFile.write(rowLines[1])
        tableFile.close()
        
if __name__ == "__main__":
    sys.exit(main())

    
