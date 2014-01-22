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
    parser.add_argument("--verbose", help="Print out detailed logging messages",
                        action = "store_true", default = False)
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
    
    args = parser.parse_args()
    if args.verbose is True:
        logging.basicConfig(level=logging.DEBUG)
        verbose = " --verbose"
    else:
        logging.basicConfig(level=logging.INFO)
        verbose = ""

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)

    trainingTrackList = TrackList(args.trainingTracksInfo)
    evalTrackList = TrackList(args.evalTracksInfo)
    checkTrackListCompatible(trainingTrackList, evalTrackList)

    sizeRange = (len(trainingTrackList), len(trainingTrackList) + 1)
    if args.allTrackCombinations is True:
        sizeRange = (1, len(trainingTrackList) + 1)

    if args.emStates is not None:
        trainFlags = "--numStates %d" % args.emStates
    else:
        trainFlags = "--supervised"
    trainFlags += " --emFac %d" % args.emFac

    #todo: try to get timing for each command
    commands = []
    rows = dict()
    for pn, pList in enumerate(subsetTrackList(trainingTrackList, sizeRange)):
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
            modPath = os.path.join(outDir,
                                   os.path.splitext(base)[0] + ".mod")
            command = "teHmmTrain.py %s %s %s %s %s" % (trainingTrackPath,
                                                        truthBed,
                                                        modPath,
                                                        verbose,
                                                        trainFlags)

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
                                                                  verbose)
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

            commands.append(command)
            
    runParallelShellCommands(commands, args.numProc)
    writeTables(args.outputDir, rows)


def subsetTrackList(trackList, sizeRange):
    """ generate tracklists of all combinations of tracks in the input list
    optionally using size range to limit the different sizes tried. so, for
    example, given input list [t1, t2, t3] and sizeRange=None this
    will gneerate [t1] [t2] [t3] [t1,t2] [t1,t3] [t2,t3] [t1,t2,t3] """
    assert sizeRange[0] > 0 and sizeRange[1] <= len(trackList) + 1
    for outLen in xrange(*sizeRange):
        for perm in itertools.combinations([x for x in xrange(len(trackList))],
                                            outLen):
            permList = TrackList()
            for trackNo in perm:
                track = copy.deepcopy(trackList.getTrackByNumber(trackNo))
                permList.addTrack(track)
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

    
