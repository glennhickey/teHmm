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

from teHmm.track import TrackData
from teHmm.hmm import MultitrackHmm
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
    
    args = parser.parse_args()
    if args.verbose is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)        
        
    # load model created with teHmmTrain.py
    logging.info("loading model %s" % args.inputModel)
    model = loadModel(args.inputModel)

    # read intervals from the bed file
    logging.info("loading target intervals from %s" % args.bedRegions)
    mergedIntervals = getMergedBedIntervals(args.bedRegions, ncol=4)
    if mergedIntervals is None or len(mergedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.bedRegions)


    # load the input
    # read the tracks, while intersecting them with the given interval
    trackData = TrackData()
    # note we pass in the trackList that was saved as part of the model
    # because we do not want to generate a new one.
    logging.info("loading tracks %s" % args.tracksInfo)
    trackData.loadTrackData(args.tracksInfo, mergedIntervals, 
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
                                               numThreads=numThreads):
        totalScore += vitLogProb
        if args.bed is not None:
            vitOutFile.write("#Viterbi Score: %f\n" % (vitLogProb))
            trackTable = trackData.getTrackTableList()[tableIndex]
            tableIndex += 1
            statesToBed(trackTable.getChrom(), trackTable.getStart(),
                        trackTable.getEnd(), vitStates, vitOutFile)

    print "Viterbi (log) score: %f" % totalScore
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
         
if __name__ == "__main__":
    sys.exit(main())
