#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import numpy as np
import math
import copy

from teHmm.track import TrackData
from teHmm.hmm import MultitrackHmm
from teHmm.cfg import MultitrackCfg
from teHmm.trackIO import getMergedBedIntervals
from teHmm.modelIO import loadModel

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate a given data set with a trained HMM. Display"
        " the log probability of the input data given the model, and "
        "optionally output the most likely sequence of hidden states.")

    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("inputModel", help="Path of hmm created with"
                        "teHmmTrain.py")
    parser.add_argument("bedRegions", help="Intervals to process")
    parser.add_argument("--bed", help="path of file to write viterbi "
                        "output to (most likely sequence of hidden states)",
                        default=None)
    parser.add_argument("--verbose", help="Print out detailed logging messages",
                        action = "store_true", default = False)
    parser.add_argument("--numThreads", help="Number of threads to use (only"
                        " applies to CFG parser for the moment)",
                        type=int, default=1)
    parser.add_argument("--slice", help="Make sure that regions are sliced"
                        " to a maximum length of the given value.  Most "
                        "useful when model is a CFG to keep memory down. "
                        "When 0, no slicing is done",
                        type=int, default=0)
    
    args = parser.parse_args()
    if args.verbose is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    if args.slice <= 0:
        args.slice = sys.maxint
        
    # load model created with teHmmTrain.py
    logging.info("loading model %s" % args.inputModel)
    model = loadModel(args.inputModel)

    # read intervals from the bed file
    logging.info("loading target intervals from %s" % args.bedRegions)
    mergedIntervals = getMergedBedIntervals(args.bedRegions, ncol=4)
    if mergedIntervals is None or len(mergedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.bedRegions)

    # slice if desired
    choppedIntervals = [x for x in slicedIntervals(mergedIntervals, args.slice)]

    # load the input
    # read the tracks, while intersecting them with the given interval
    trackData = TrackData()
    # note we pass in the trackList that was saved as part of the model
    # because we do not want to generate a new one.
    logging.info("loading tracks %s" % args.tracksInfo)
    trackData.loadTrackData(args.tracksInfo, choppedIntervals, 
                            model.getTrackList())

    # do the viterbi algorithm
    if isinstance(model, MultitrackHmm):
        logging.info("running viterbi algorithm")
    elif isinstance(model, MultitrackCfg):
        logging.info("running CYK algorithm")        

    if args.bed is not None:
        vitOutFile = open(args.bed, "w")
    totalScore = 0
    tableIndex = 0
    for vitLogProb, vitStates in model.viterbi(trackData,
                                               numThreads=args.numThreads):
        totalScore += vitLogProb
        if args.bed is not None:
            vitOutFile.write("#Viterbi Score: %f\n" % (vitLogProb))
            trackTable = trackData.getTrackTableList()[tableIndex]
            tableIndex += 1
            statesToBed(trackTable.getChrom(), trackTable.getStart(),
                        trackTable.getEnd(), vitStates, vitOutFile)

    print "Viterbi (log) score: %f" % totalScore
    if isinstance(model, MultitrackHmm) and model.current_iteration is not None:
        print "Number of EM iterations: %d" % model.current_iteration
    if args.bed is not None:
        vitOutFile.close()

def statesToBed(chrom, start, end, states, bedFile):
    """write a sequence of states out in bed format where intervals are
    maximum runs of contiguous states."""
    assert len(states) == end - start
    prevInterval = (chrom, start, start + 1, states[0])
    for i in xrange(1, len(states) + 1):
        if i < len(states):
            state = states[i]
        else:
            state = None
        if state != prevInterval[3]:
            assert prevInterval[3] is not None
            assert prevInterval[1] >= start and prevInterval[2] <= end
            bedFile.write("%s\t%d\t%d\t%s\n" % prevInterval)
            prevInterval = (prevInterval[0], prevInterval[2],
                            prevInterval[2] + 1, state)
        else:
            prevInterval = (prevInterval[0], prevInterval[1],
                            prevInterval[2] + 1, prevInterval[3])

def slicedIntervals(bedIntervals, chunkSize):
    """slice bed intervals by a given length.  used as a quick way to get
    cfg working via cutting up the input beds (after they get merged)."""
    for interval in bedIntervals:
        iLen = interval[2] - interval[1]
        if iLen <= chunkSize:
            yield interval
        else:
            nCuts = int(math.ceil(float(iLen) / float(chunkSize)))
            for sliceNo in xrange(nCuts):
                sInt = list(copy.deepcopy(interval))
                sInt[1] = sliceNo * chunkSize
                if sliceNo < nCuts - 1:
                    sInt[2] = sInt[1] + chunkSize
                assert sInt[2] > sInt[1]
                yield tuple(sInt)
         
if __name__ == "__main__":
    sys.exit(main())
