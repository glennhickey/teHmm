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
    from teHmm.parameterAnalysis import hierarchicalCluster, rankHierarchies
    from teHmm.parameterAnalysis import pcaFlatten, plotPoints2d
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
    parser.add_argument("--pca", help="Print emission pca scatters"
                        " to given file in PDF format", default=None)
    parser.add_argument("--t", help="Print transition matrix to given"
                        " file in GRAPHVIZ DOT format.  Convert to PDF with "
                        " dot <file> -Tpdf > <outFile>", default=None)
    
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

    if args.pca is not None:
        if canPlot is False:
            raise RuntimeError("Unable to write plots.  Maybe matplotlib is "
                               "not installed?")
        writeEmissionScatters(model, args)

    if args.t is not None:
        writeTransitionGraph(model, args)


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
    # list for each state
    allPoints = []
    for i in xrange(N):
        allPoints.append([])

    for track in trackList:
        hcNames.append(track.getName())
        points = [emissionDist[track.getNumber()][x] for x in xrange(N)]
        for j in xrange(N):
            allPoints[j] += list(points[j])
        hc = hierarchicalCluster(points, normalizeDistances=True)
        hcList.append(hc)

    # all at once
    hc = hierarchicalCluster(allPoints, normalizeDistances=True)
    hcList.append(hc)
    hcNames.append("all_tracks")
    
    # write clusters to pdf (ranked in decreasing order based on total
    # branch length)
    ranks = rankHierarchies(hcList)
    plotHierarchicalClusters([hcList[i] for i in ranks],
                             [hcNames[i] for i in ranks],
                             stateNames, args.ec)

def writeEmissionScatters(model, args):
    """ print a pca scatterplot of states for each track (where each
    state is a point represented by its distribution for that track)"""
    trackList = model.getTrackList()
    stateNameMap = model.getStateNameMap()
    emission = model.getEmissionModel()
    # [TRACK][STATE][SYMBOL]
    emissionDist = np.exp(emission.getLogProbs())

    # leaf names of our clusters are the states
    stateNames = map(stateNameMap.getMapBack, xrange(len(stateNameMap)))
    N = len(stateNames)

    # scatter for each track
    scatterList = []
    scatterScores = []
    hcNames = []
        
    for track in trackList:
        hcNames.append(track.getName())
        points = [emissionDist[track.getNumber()][x] for x in xrange(N)]
        try:
            pcaPoints, score = pcaFlatten(points)
            scatterList.append(pcaPoints)
            scatterScores.append(score)
        except Exception as e:
            print "PCA FAIL %s" % track.getName()

    # sort by score
    zipRank = zip(scatterScores, [x for x in xrange(len(scatterScores))])
    zipRank = sorted(zipRank)
    ranking = zip(*zipRank)[1]

    if len(scatterList) > 0:
        plotPoints2d([scatterList[i] for i in ranking],
                     [hcNames[i] for i in ranking],
                     stateNames, args.pca)    

def writeTransitionGraph(model, args):
    """ write a graphviz text file """
    trackList = model.getTrackList()
    stateNameMap = model.getStateNameMap()
    stateNames = map(stateNameMap.getMapBack, xrange(len(stateNameMap)))
    stateNames = map(lambda x: x.replace("-", "_"), stateNames)
    stateNames = map(lambda x: x.replace("|", "_"), stateNames)
    
    N = len(stateNames)
    f = open(args.t, "w")
    f.write("Digraph G {\n")
    for i, state in enumerate(stateNames):
        for j, toState in enumerate(stateNames):
            tp = model.getTransitionProbs()[i, j]
            if tp > 0. and i != j:
                label = "label=\"%.2f\"" % (tp * 100.)
                width = "penwidth=%d" % (1 + int(tp / 20))
                f.write("%s -> %s [%s,%s];\n" % (state, toState, label, width))
    f.write("}\n")
    f.close()
    
if __name__ == "__main__":
    sys.exit(main())
