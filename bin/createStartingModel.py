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

from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import getLocalTempPath
from teHmm.track import TrackList, Track, CategoryMap
from teHmm.trackIO import readBedIntervals, getMergedBedIntervals

def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create starting transition and emission distributions "
        "from a candidate BED annotation, which can"
        " be used with teHmmTrain.py using the --initTransProbs and "
        "--initEmProbs options, respectively.  The distributions created here"
        " are extremely simple, but this can be a good shortcut to at least "
        "getting the state names into the init files, which can be further "
        "tweeked by hand.")

    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("trackName", help="Name of Track to use as initial"
                        " annotation")
    parser.add_argument("queryBed", help="Bed file with regions to query")
    parser.add_argument("outTransProbs", help="File to write transition model"
                        " to")
    parser.add_argument("outEmProbs", help="File to write emission model to")
    parser.add_argument("--numOut", help="Number of \"outside\" states to add"
                        " to the model.", default=1, type=int)
    parser.add_argument("--outName", help="Name of outside states (will have"
                        " numeric suffix if more than 1)", default="Outside")
    parser.add_argument("--mode", help="Strategy for initializing the "
                        "transition graph: {\'star\': all states are connected"
                        " to the oustide state(s) but not each other; "
                        " \'data\': transitions estimated from input bed; "
                        " \'full\': dont write edges and let teHmmTrain.py "
                        "initialize as a clique}", default="star")
    parser.add_argument("--selfTran", help="This script will always write all"
                        " the self-transition probabilities to the output file. "
                        "They will all be set to the specified value using this"
                        " option, or estimated from the data if -1", default=-1.,
                        type=float)
    parser.add_argument("--em", help="Emission probability for input track ("
                        "ie probability that state emits itself)",
                        type=float, default=0.95)
                        
    addLoggingOptions(parser)
    args = parser.parse_args()
    if args.mode == "star" and args.numOut < 1:
        raise RuntimeError("--numOut must be at least 1 if --mode star is used")
    if args.mode != "star" and args.mode != "data" and args.mode != "full":
        raise RuntimeError("--mode must be one of {star, data, full}")
    if args.mode == "data":
        raise RuntimeError("--data not implemented yet")
    assert os.path.isfile(args.tracksInfo)
    
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    # Read the tracks info
    trackList = TrackList(args.tracksInfo)
    # Extract the track we want
    track = trackList.getTrackByName(args.trackName)
    if track is None:
        raise RuntimeError("Track %s not found in tracksInfo" % args.trackName)
    trackPath = track.getPath()
    if track.getDist() != "multinomial":
        raise RuntimeError("Track %s does not have multinomial distribution" %
                           args.trackName)
    if track.getScale() is not None or track.getLogScale() is not None:
        raise RuntimeError("Track %s must not have scale" % args.trackName)
    
    # read query intervals from the bed file
    logger.info("loading query intervals from %s" % args.queryBed)
    mergedIntervals = getMergedBedIntervals(args.queryBed, ncol=4)
    if mergedIntervals is None or len(mergedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.queryBed)

    # read the track, while intersecting with query intervals
    # (track is saved as temp XML file for sake not changing interface)
    bedIntervals = []
    for queryInterval in mergedIntervals:
        bedIntervals += readBedIntervals(trackPath,
                                        ncol = track.getValCol() + 1,
                                        chrom=queryInterval[0],
                                        start=queryInterval[1],
                                        end=queryInterval[2])

    # 1st pass to collect set of names
    nameMap = CategoryMap(reserved = 0)
    for interval in bedIntervals:
        nameMap.update(interval[track.getValCol()])
    outNameMap = CategoryMap(reserved = 0)
    for i in xrange(args.numOut):
        outName = args.outName
        if args.numOut > 1:
            outName += str(i)
        assert nameMap.has(outName) is False
        outNameMap.update(outName)

    # write the transition model for use with teHmmTrain.py --initTransProbs    
    writeTransitions(bedIntervals, nameMap, outNameMap, args)

    # write the emission model for use with teHmmTrain.py --initEmProbs
    writeEmissions(bedIntervals, nameMap, outNameMap, args)


def writeTransitions(bedIntervals, nameMap, outNameMap, args):
    tfile = open(args.outTransProbs, "w")
    
    # do the self transitions
    N = len(nameMap)
    selfTran = args.selfTran + np.zeros((N))
    if args.selfTran < 0:
        tot = np.zeros((N))
        num = np.zeros((N))
        for interval in bedIntervals:
            assert nameMap.has(interval[3])
            state = nameMap.getMap(interval[3])
            assert state < N
            num[state] += 1
            tot[state] += interval[2] - interval[1] - 1
        selfTran = tot / (tot + num)

    for state, i in nameMap.catMap.items():
        tfile.write("%s\t%s\t%f\n" % (state, state, selfTran[i]))
        if args.mode == "star":
            outTrans = (1. - selfTran[i]) / float(args.numOut)
            for outState, j in outNameMap.catMap.items():
                tfile.write("%s\t%s\t%f\n" % (state, outState, outTrans))

    # do the outside states
    if args.numOut > 0:
        outselfTran = args.selfTran + np.zeros((args.numOut))
        if args.selfTran < 0:
            # hack for now (should be from data above)
            logger.debug("Hacky maximum used for outside state self transition")
            outselfTran = max(selfTran) + np.zeros((args.numOut))
            
        for state, i in outNameMap.catMap.items():
            tfile.write("%s\t%s\t%f\n" % (state, state, outselfTran[i]))
                
    tfile.close()
            

def writeEmissions(bedIntervals, nameMap, outNameMap, args):
    efile = open(args.outEmProbs, "w")

    for state, i in nameMap.catMap.items():
        efile.write("%s\t%s\t%s\t%f\n" % (state, args.trackName, state,
                                          args.em))
    for state, i in outNameMap.catMap.items():
        efile.write("%s\t%s\t%s\t%f\n" % (state, args.trackName, "__NoNE__",
                                          args.em))
    
    efile.close()

if __name__ == "__main__":
    sys.exit(main())
