#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import numpy as np

from teHmm.track import TrackData
from teHmm.hmm import MultitrackHmm
from teHmm.cfg import MultitrackCfg
from teHmm.modelIO import loadModel
try:
    from teHmm.parameterAnalysis import plotHierarchicalClusters
    from teHmm.parameterAnalysis import hierarchicalCluster
    canPlot = True
except:
    canPlot = False

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Print out paramaters of a teHMM")

    parser.add_argument("inputModel", help="Path of teHMM model created with"
                        " teHmmTrain.py")
    parser.add_argument("--nameMap", help="Print out name map tables",
                        action="store_true", default=False)
    parser.add_argument("--ec", help="Print emission distribution clusterings"
                        " to given file in PDF format", default=None)
    
    args = parser.parse_args()

    # load model created with teHmmTrain.py
    model = loadModel(args.inputModel)

    # crappy print method
    print model

    if args.nameMap is True:
        print "State Maps:"
        trackList = model.trackList
        if trackList is None:
            print "TrackList: None"
        else:
            for track in trackList:
                print "Track: %s" % track.getName()
                print " map %s " % track.getValueMap().catMap
                print " pam %s " % track.getValueMap().catMapBack

    if args.ec is not None:
        if canPlot is False:
            raise RuntimeError("Unable to write plots.  Maybe matplotlib is "
                               "not installed?")
        writeEmissionClusters(model, args)

def writeEmissionClusters(model, args):
    """ print a hierachical clustering of states for each track (where each
    state is a point represented by its distribution for that track)"""
    trackList = model.getTrackList()
    stateNameMap = model.getStateNameMap()
    emission = model.getEmissionModel()
    # [TRACK][STATE][SYMBOL]
    emissionDist = np.exp(emission.getLogProbs())

    # leaf names of our clusters are the states
    stateNames = map(stateNameMap.getMapBack, xrange(len(stateNameMap)))
    N = len(stateNames)

    # cluster for each track
    hcList = []
    hcNames = []
    for track in trackList:
        hcNames.append(track.getName())
        points = [emissionDist[track.getNumber()][x] for x in xrange(N)]
        hc = hierarchicalCluster(points, normalizeDistances=True)
        hcList.append(hc)

    # write clusters to pdf
    plotHierarchicalClusters(hcList, hcNames, stateNames, args.ec)
        
    #todo: ranking
    
    
if __name__ == "__main__":
    sys.exit(main())
