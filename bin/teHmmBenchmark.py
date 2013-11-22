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

def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train, evalaute, then compare hmm model on input")

    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
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
    
    args = parser.parse_args()
    if args.verbose is True:
        logging.basicConfig(level=logging.DEBUG)
        verbose = " --verbose"
    else:
        logging.basicConfig(level=logging.INFO)
        verbose = ""

    if not os.path.exists(args.outputDir):
        runShellCommand("mkdir %s" % args.outputDir)

    inputTrackList = TrackList(args.tracksInfo)

    sizeRange = (len(inputTrackList), len(inputTrackList) + 1)
    if args.allTrackCombinations is True:
        sizeRange = (1, len(inputTrackList) + 1)

    if args.emStates is not None:
        trainFlags = "--numStates %d" % args.emStates
    else:
        trainFlags = "--supervised"

    #todo: try to get timing for each command
    commands = []

    for pn, pList in enumerate(permuteTrackList(inputTrackList, sizeRange)):
        if len(pList) == len(inputTrackList):
            outDir = args.outputDir
        else:
            outDir = os.path.join(args.outputDir, "perm%d" % pn)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        trackPath = os.path.join(outDir, "tracks.xml")
        pList.saveXML(trackPath)
        
        for inBed in args.inBeds:
            # train
            base = os.path.basename(inBed)
            modPath = os.path.join(outDir,
                                   os.path.splitext(base)[0] + ".mod")
            command = "teHmmTrain.py %s %s %s %s %s" % (trackPath,
                                                        inBed,
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
            command += " && teHmmEval.py %s %s %s --bed %s %s" % (trackPath,
                                                                  modPath,
                                                                  inBed,
                                                                  evalBed,
                                                                  verbose)
            # compare
            compPath = os.path.join(outDir,
                                    os.path.splitext(base)[0] + "_comp.txt")
            command += " && compareBedStates.py %s %s > %s" % (inBed,
                                                               evalBed,
                                                               compPath)
            commands.append(command)

    runParallelShellCommands(commands, args.numProc)


def permuteTrackList(trackList, sizeRange):
    """ generate tracklists of all combinations of tracks in the input list
    optionally using size range to limit the different sizes tried. so, for
    example, given input list [t1, t2, t3] and sizeRange=None this
    will gneerate [t1] [t2] [t3] [t1,t2] [t1,t3] [t2,t3] [t1,t2,t3] """
    assert sizeRange[0] > 0 and sizeRange[1] <= len(trackList) + 1
    for outLen in xrange(*sizeRange):
        for perm in itertools.permutations([x for x in xrange(outLen)]):
            permList = TrackList()
            for trackNo in perm:
                track = copy.deepcopy(trackList.getTrackByNumber(trackNo))
                permList.addTrack(track)
            yield permList
    
if __name__ == "__main__":
    sys.exit(main())

    
