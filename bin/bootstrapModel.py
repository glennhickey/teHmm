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
from teHmm.modelIO import loadModel

def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create starting transition and emission distributions "
        " that be used with teHmmTrain.py using the --initTransProbs and "
        "--initEmProbs options, respectively.  The distributions will be "
        " derived by an already-trained model.  This tool is written to"
        " allow combining supervised and unsupervised training.  IE a "
        " supervised model is created (teHmmTrain.py with --supervised "
        " option).  This tool can then be used to create the necessary "
        " files to bootstrap an unsupervised training run with a subset"
        " of the parameters.")

    parser.add_argument("inputModel", help="Path of input model to use "
                        "for bootstrap parameter creation")
    parser.add_argument("outTransProbs", help="File to write transition model"
                        " to (for use with teHmmTrain.py --initTransProbs and"
                        " --forceTransProbs)")
    parser.add_argument("outEmProbs", help="File to write emission model to"
                        " (for use with teHmmTrain.py --initEmProbs and "
                        " --forceEmProbs)")
    parser.add_argument("--ignore", help="comma-separated list of states to ignore from"
                        " inputModel", default=None)
    parser.add_argument("--numAdd", help="Number of \"unlabeled\" states to add"
                        " to the model.", default=0, type=int)
    parser.add_argument("--numTotal", help="Add unlabeled states such that"
                        " output model has given number of states.  If input "
                        "model already has a greater number of states then"
                        " none added", default=0, type=int)
    parser.add_argument("--stp", help="Self-transition probality assigned to"
                        " added states.", default=0.9, type=float)
                        
    addLoggingOptions(parser)
    args = parser.parse_args()
    if args.numAdd != 0 and args.numTotal != 0:
        raise RuntimeError("--numAdd and --numTotal mutually exclusive")

    # load model created with teHmmTrain.py
    logger.info("loading model %s" % args.inputModel)
    model = loadModel(args.inputModel)

    # parse ignore states
    if args.ignore is None:
        args.ignore = set()
    else:
        args.ignore = set(args.ignore.split(","))

    # make sure we have a state name for every state (should really
    # be done within hmm...)
    stateMap = model.getStateNameMap()
    if stateMap is None:
        stateMap = CategoryMap(reserved = 0)
        for i in xrange(model.getEmissionModel().getNumStates()):
            stateMap.getMap(str(i), update=True)

    # write the transition probabilities
    writeTransitions(model, stateMap, args)

    # write the emission probabilities
    writeEmissions(model, stateMap, args)


def writeTransitions(hmm, stateMap, args):
    """ write the transitions (in the crappy text file format) from the hmm
    """
    tfile = open(args.outTransProbs, "w")

    em = hmm.getEmissionModel()
    tranProbs = hmm.getTransitionProbs()
    numStates = len(tranProbs)
    assert len(stateMap) == numStates
    count = 0
    
    for fromStateNo in xrange(numStates):
        fromName = stateMap.getMapBack(fromStateNo)
        assert fromName is not None
        if fromName not in args.ignore:
            count += 1
            for toStateNo in xrange(numStates):
                toName = stateMap.getMapBack(toStateNo)
                assert toName is not None
                if toName not in args.ignore:
                    tfile.write("%s\t%s\t%e\n" % (fromName, toName,
                                                   tranProbs[fromStateNo,
                                                             toStateNo]))

    # write additional states
    numAdd = args.numAdd
    if args.numTotal > count:
        numAdd = args.numTotal - count
    for i in xrange(numAdd):
        name = "unlabled%d" % i
        if stateMap.has(name) is True:
            S = string.ascii_uppercase + string.digits
            tag = ''.join(random.choice(S) for x in range(5))
            name += "_%s" % tag
        assert stateMap.has(name) is False
        tfile.write("%s\t%s\t%e\n" % (name, name, args.stp))

    tfile.close()
            

def writeEmissions(hmm, stateMap, args):
    """ write the emissions (in the crappy text file format) from the hmm
    """
    efile = open(args.outEmProbs, "w")
    em = hmm.getEmissionModel()
    emProbs = np.exp(em.getLogProbs())
    trackList = hmm.getTrackList()
    assert len(trackList) > 0
    numStates = len(emProbs[0])
    assert len(stateMap) == numStates
    assert len(trackList) == len(emProbs)

    for stateNo in xrange(numStates):
        stateName = stateMap.getMapBack(stateNo)
        for track in trackList:
            trackName = track.getName()
            trackNo = track.getNumber()
            # GAUSSIAN : TRACK STATE MEAN STDEV
            if track.getDist() == "gaussian":
                mean, stdev = em.getGaussianParams(trackNo, stateNo)
                efile.write("%s\t%s\t%e\t%e\n" % (trackName, stateName,
                                                  mean, stdev))

            # BINARY : TRACK STATE 1 Pr[not None]
            elif track.getDist() == "binary":
                efile.write("%s\t%s\t%s\t%e\n" % (trackName, stateName,"0",
                                                  emProbs[trackNo][stateNo][1]))
                efile.write("%s\t%s\t%s\t%e\n" % (trackName, stateName,"1",
                                                  emProbs[trackNo][stateNo][2]))

            # ANYTHING ELSE : TRACK STATE SYMBOL PR[SYMBOL]
            else:
                catMap = track.getValueMap()
                if catMap.getReserved() == 2:
                    efile.write("%s\t%s\t%s\t%e\n" % (trackName, stateName,
                                                      "None",
                                                      emProbs[trackNo][stateNo][1]))
                for symbolName, symbolId in catMap.catMap.items():
                    efile.write("%s\t%s\t%s\t%e\n" % (trackName, stateName,
                                                      symbolName,
                                                      emProbs[trackNo][stateNo][symbolId]))
                                                  
    efile.close()

if __name__ == "__main__":
    sys.exit(main())
