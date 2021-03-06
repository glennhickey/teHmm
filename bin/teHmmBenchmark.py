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
    parser.add_argument("outputDir", help="directory to write output")
    parser.add_argument("inBeds", nargs="*", help="list of training beds")
    parser.add_argument("--evalTracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks used"
                        " for evaluation (only need if different from"
                        " trainingTracksInfo", default=None)
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
    parser.add_argument("--fixStart", help="Do not learn start parameters"
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
    parser.add_argument("--emRandRange", help="When randomly initialzing a"
                        " multinomial emission distribution, constrain"
                        " the values to the given range (pair of "
                        "comma-separated numbers).  Overridden by "
                        "--initEmProbs and --forceEmProbs when applicable."
                        " Completely overridden by --flatEm (which is equivalent"
                        " to --emRandRange .5,.5.). Actual values used will"
                        " always be normalized.", default=None)    
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
    parser.add_argument("--segment", help="Input bed files are also used to "
                        "segment data.  Ie teHmmTrain is called with --segment"
                        " set to the input file. Not currently working with "
                        " --supervised",
                        action = "store_true", default=False)
    parser.add_argument("--segLen", help="Effective segment length used for"
                        " normalizing input segments (specifying 0 means no"
                        " normalization applied) in training", type=int,
                        default=None)
    parser.add_argument("--truth", help="Use specifed file instead of "
                        "input file(s) for truth comparison.  Makes sense"
                        " when --segment is specified and only one input"
                        " bed specified", default = None)
    parser.add_argument("--eval", help="Bed file used for evaluation.  It should"
                        " cover same region in same order as --truth.  Option "
                        "exists mostly to specify segmentation of --truth",
                        default=None)
    parser.add_argument("--seed", help="Seed for random number generator"
                        " which will be used to initialize emissions "
                        "(if --flatEM and --supervised not specified)",
                        default=None, type=int)
    parser.add_argument("--reps", help="Number of training replicates (with "
                        " different"
                         " random initializations) to run. The replicate"
                         " with the highest likelihood will be chosen for the"
                         " output", default=None, type=int)
    parser.add_argument("--numThreads", help="Number of threads to use when"
                        " running training replicates (see --rep) in parallel.",
                        type=int, default=None)
    parser.add_argument("--emThresh", help="Threshold used for convergence"
                        " in baum welch training.  IE delta log likelihood"
                        " must be bigger than this number (which should be"
                        " positive) for convergence", type=float,
                        default=None)
    parser.add_argument("--fit", help="Run fitStateNames.py to automap names"
                        " before running comparison", action="store_true",
                        default=False)
    parser.add_argument("--fitOpts", help="Options to pass to fitStateNames.py"
                        " (only effective if used with --fit)", default=None)
    parser.add_argument("--saveAllReps", help="Save all replicates (--reps)"
                        " models to disk, instead of just the best one"
                        ". Format is <outputModel>.repN.  There will be "
                        " --reps -1 such models saved as the best output"
                        " counts as a replicate.  Comparison statistics"
                        " will be generated for each rep.",
                        action="store_true", default=False)
    parser.add_argument("--maxProb", help="Gaussian distributions and/or"
                        " segment length corrections can cause probability"
                        " to *decrease* during BW iteration.  Use this option"
                        " to remember the parameters with the highest probability"
                        " rather than returning the parameters after the final "
                        "iteration.", action="store_true", default=False)
    parser.add_argument("--maxProbCut", help="Use with --maxProb option to stop"
                        " training if a given number of iterations go by without"
                        " hitting a new maxProb", default=None, type=int)
    parser.add_argument("--transMatEpsilons", help="By default, epsilons are"
                        " added to all transition probabilities to prevent "
                        "converging on 0 due to rounding error only for fully"
                        " unsupervised training.  Use this option to force this"
                        " behaviour for supervised and semisupervised modes",
                        action="store_true", default=False)

        
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    logOps = "--logLevel %s" % getLogLevelString()
    if args.logFile is not None:
        logOps += " --logFile %s" % args.logFile

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
    if args.evalTracksInfo is None:
        args.evalTracksInfo = args.trainingTracksInfo

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
    if args.emRandRange is not None:
        trainFlags += " --emRandRange %s" % args.emRandRange
    if args.segLen is not None:
        trainFlags += " --segLen %d" % args.segLen
    if args.seed is not None:
        trainFlags += " --seed %d" % args.seed
    if args.reps is not None:
        trainFlags += " --reps %d" % args.reps
    if args.numThreads is not None:
        trainFlags += " --numThreads %d" % args.numThreads
    if args.emThresh is not None:
        trainFlags += " --emThresh %f" % args.emThresh
    if args.saveAllReps is True:
        trainFlags += " --saveAllReps"
    if args.maxProb is True:
        trainFlags += " --maxProb"
    if args.transMatEpsilons is True:
        trainFlags += " --transMatEpsilons"
    if args.maxProbCut is not None:
        trainFlags += " --maxProbCut %d" % args.maxProbCut

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
        for maskTrack in trainingTrackList.getMaskTracks():
            pList.addTrack(copy.deepcopy(maskTrack))
        pList.saveXML(trainingTrackPath)
        epList = TrackList()
        for track in pList:
            t = copy.deepcopy(evalTrackList.getTrackByName(track.getName()))
            epList.addTrack(t)
        for maskTrack in trainingTrackList.getMaskTracks():
            epList.addTrack(copy.deepcopy(maskTrack))
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
                if args.segment is True:
                    command += " --segment %s" % truthBed

            # view
            viewPath = os.path.join(outDir,
                                   os.path.splitext(base)[0] + "_view.txt")
            command += " && teHmmView.py %s > %s" % (modPath, viewPath)

            # evaluate
            numReps = 1
            if args.reps is not None and args.saveAllReps is True:
                numReps = args.reps
                assert numReps > 0
            missed = 0
            # little hack to repeat evaluation for each training replicate
            for repNum in xrange(-1, numReps-1):
                if repNum == -1:
                    repSuffix = ""
                else:
                    repSuffix = ".rep%d" % repNum                
                evalBed = os.path.join(outDir,
                                       os.path.splitext(base)[0] + "_eval.bed" +
                                       repSuffix)
                hmmEvalInputBed = testBed
                if args.eval is not None:
                    hmmEvalInputBed = args.eval
                bicPath = os.path.join(outDir,
                                       os.path.splitext(base)[0] + "_bic.txt" +
                                       repSuffix)

                command += " && teHmmEval.py %s %s %s --bed %s %s --bic %s" % (
                    evalTrackPath,
                    modPath + repSuffix,
                    hmmEvalInputBed,
                    evalBed,
                    logOps,
                    bicPath)
                zin = True

                if args.segment is True:
                    command += " --segment"

                # fit
                compTruth = testBed
                if args.truth is not None:
                    compTruth = args.truth
                compareInputBed = evalBed
                if args.fit is True:
                    fitBed = os.path.join(outDir,
                                          os.path.splitext(base)[0] + "_eval_fit.bed" +
                                          repSuffix)
                    command += " && fitStateNames.py %s %s %s --tl %s" % (compTruth,
                                                                          evalBed,
                                                                          fitBed,
                                                                          evalTrackPath)
                    if args.fitOpts is not None:
                        command += " " + args.fitOpts
                    compareInputBed = fitBed

                # compare
                compPath = os.path.join(outDir,
                                        os.path.splitext(base)[0] + "_comp.txt" +
                                        repSuffix)
                command += " && compareBedStates.py %s %s --tl %s > %s" % (
                    compTruth,
                    compareInputBed,
                    evalTrackPath,
                    compPath)
            

                # make table row
                if repSuffix == "":
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

    
