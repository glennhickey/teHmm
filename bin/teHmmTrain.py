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
from teHmm.track import CategoryMap

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

    catMap = None
    # state number is overrided by the input bed file in supervised mode
    if args.supervised is True:
        catMap = mapStateNames(bedIntervals)
        args.numStates = len(catMap)
        
    # create the independent emission model
    numSymbolsPerTrack = trackData.getNumSymbolsPerTrack()
    emissionModel = IndependentMultinomialEmissionModel(args.numStates,
                                                        numSymbolsPerTrack)

    # create the hmm
    hmm = MultitrackHmm(emissionModel, n_iter=args.iter, state_name_map=catMap)

    # do the training
    if args.supervised is False:
        hmm.train(trackData)
    else:
        hmm.supervisedTrain(trackData, bedIntervals)

    # write the model to a pickle
    hmm.save(args.outputModel)


def mapStateNames(bedIntervals):
    """ sanitize the states (column 4) of each bed interval, mapping to unique
    integer.  return the map"""
    catMap = CategoryMap()
    for interval in bedIntervals:
        if len(interval) < 4 or interval[3] is None:
            raise RuntimeError("Could not read state from 4th column" %
                               str(interval))
        catMap.update(interval[3])
    return catMap
    
if __name__ == "__main__":
    sys.exit(main())

    
