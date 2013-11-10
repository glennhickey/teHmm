#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse

from teHmm.track import TrackData
from teHmm.trackIO import readBedIntervals
from teHmm.hmm import MultitrackHmm
from teHmm.emission import IndependentMultinomialEmissionModel

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create a teHMM")

    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("trainingBed", help="Path of BED file containing"
                        " elements to train  model on")
    parser.add_argument("outputModel", help="Path of output hmm")
    parser.add_argument("--numStates", help="Number of states in model",
                        type = int, default=2)
    parser.add_argument("--iter", help="Number of EM iterations",
                        type = int, default=1000)
    parser.add_argument("--supervised", help="Use name (4th) column of "
                        "<traingingBed> for the true hidden states of the"
                        " model.  Transition parameters will be estimated"
                        " directly from this information rather than EM."
                        " NOTE: The number of states will be determined "
                        "from the bed.  States must be labled 0,1,2 etc.",
                        action = "store_true", default = False)
    
    args = parser.parse_args()

    # read training intervals from the bed file
    bedIntervals = readBedIntervals(args.trainingBed, ncol=4)
    if bedIntervals is None or len(bedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.trainingBed)
    # read the tracks, while intersecting them with the training intervals
    trackData = TrackData()
    trackData.loadTrackData(args.tracksInfo, bedIntervals)

    # state number is overrided by the input bed file in supervised mode
    if args.supervised is True:
        print "C"
        args.numStates = countStates(bedIntervals)
        
    # create the independent emission model
    numSymbolsPerTrack = trackData.getNumSymbolsPerTrack()
    emissionModel = IndependentMultinomialEmissionModel(args.numStates,
                                                        numSymbolsPerTrack)

    # create the hmm
    hmm = MultitrackHmm(emissionModel, n_iter=args.iter)

    # do the training
    if args.supervised is False:
        hmm.train(trackData)
    else:
        hmm.supervisedTrain(trackData, bedIntervals)

    # write the model to a pickle
    hmm.save(args.outputModel)


def countStates(bedIntervals):
    """ check the supervised training states. return number of unique states"""
    stateSet = set()
    maxElem = 0
    minElem = len(bedIntervals)
    for interval in bedIntervals:
        try:
            stateVal = int(interval[3])
        except:
            raise RuntimeError("Invalid bed value %s found.  Supervised states"
                               " must be integers" % interval[3])
        stateSet.add(stateVal)
        maxElem = max(maxElem, stateVal)
        minElem = min(minElem, stateVal)
    if minElem != 0 or maxElem != len(stateSet) - 1:
        raise RuntimeError("Supervised training states must be integers"
                           " from 0 to N (with no missing values in between"
                           ". instead we got %d different values in range"
                           "[%d, %d]" % (len(stateSet), minElem, maxElem))
    return len(stateSet)
           
    
if __name__ == "__main__":
    sys.exit(main())

    
