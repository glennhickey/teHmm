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
    
    args = parser.parse_args()

    # read training intervals from the bed file
    bedIntervals = readBedIntervals(args.trainingBed)
    if bedIntervals is None or len(bedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.trainingBed)
    # read the tracks, while intersecting them with the training intervals
    trackData = TrackData()
    trackData.loadTrackData(args.tracksInfo, bedIntervals)

    # create the independent emission model
    numSymbolsPerTrack = trackData.getNumSymbolsPerTrack()
    emissionModel = IndependentMultinomialEmissionModel(args.numStates,
                                                        numSymbolsPerTrack)

    # create the hmm
    hmm = MultitrackHmm(emissionModel, n_iter=args.iter)

    # do the training
    hmm.train(trackData)

    # write the model to a pickle
    hmm.save(args.outputModel)
     
if __name__ == "__main__":
    sys.exit(main())
