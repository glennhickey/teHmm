#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging

from teHmm.track import TrackData
from teHmm.trackIO import readBedIntervals, getMergedBedIntervals
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
                        type = int, default=10)
    parser.add_argument("--supervised", help="Use name (4th) column of "
                        "<traingingBed> for the true hidden states of the"
                        " model.  Transition parameters will be estimated"
                        " directly from this information rather than EM."
                        " NOTE: The number of states will be determined "
                        "from the bed.  States must be labled 0,1,2 etc.",
                        action = "store_true", default = False)
    parser.add_argument("--verbose", help="Print out detailed logging messages",
                        action = "store_true", default = False)

    
    args = parser.parse_args()
    if args.verbose is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)        

    # read training intervals from the bed file
    logging.info("loading training intervals from %s" % args.trainingBed)
    mergedIntervals = getMergedBedIntervals(args.trainingBed, ncol=4)
    if mergedIntervals is None or len(mergedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.trainingBed)

    # read the tracks, while intersecting them with the training intervals
    logging.info("loading tracks %s" % args.tracksInfo)
    trackData = TrackData()
    trackData.loadTrackData(args.tracksInfo, mergedIntervals)

    catMap = None
    truthIntervals = None
    # state number is overrided by the input bed file in supervised mode
    if args.supervised is True:
        logging.info("processing supervised state names")
        # we reload because we don't want to be merging them here
        truthIntervals = readBedIntervals(args.trainingBed, ncol=4)
        catMap = mapStateNames(truthIntervals)
        args.numStates = len(catMap)
        
    # create the independent emission model
    logging.info("creating emission model")
    numSymbolsPerTrack = trackData.getNumSymbolsPerTrack()
    emissionModel = IndependentMultinomialEmissionModel(args.numStates,
                                                        numSymbolsPerTrack)

    # create the hmm
    logging.info("creating hmm model")
    hmm = MultitrackHmm(emissionModel, n_iter=args.iter, state_name_map=catMap)

    # do the training
    if args.supervised is False:
        logging.info("training via EM")
        hmm.train(trackData)
    else:
        logging.info("training from input bed states")
        hmm.supervisedTrain(trackData, truthIntervals)

    # write the model to a pickle
    logging.info("saving trained model to %s" % args.outputModel)
    hmm.save(args.outputModel)


def mapStateNames(bedIntervals):
    """ sanitize the states (column 4) of each bed interval, mapping to unique
    integer in place.  return the map"""
    catMap = CategoryMap(reserved=0)
    for idx, interval in enumerate(bedIntervals):
        if len(interval) < 4 or interval[3] is None:
            raise RuntimeError("Could not read state from 4th column" %
                               str(interval))
        bedIntervals[idx] = (interval[0], interval[1], interval[2],
                             catMap.getMap(interval[3], update=True))
    return catMap
    
if __name__ == "__main__":
    sys.exit(main())

    
