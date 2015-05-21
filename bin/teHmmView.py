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
from teHmm.common import EPSILON
try:
    from teHmm.parameterAnalysis import plotHierarchicalClusters
    from teHmm.parameterAnalysis import hierarchicalCluster, rankHierarchies
    from teHmm.parameterAnalysis import pcaFlatten, plotPoints2d, plotHeatMap
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
    parser.add_argument("--ecn", help="Like --ec option but only print non"
                        " numeric tracks", default=None)
    parser.add_argument("--pca", help="Print emission pca scatters"
                        " to given file in PDF format", default=None)
    parser.add_argument("--hm", help="Print heatmap of emission distribution means"
                        " for (only) numeric tracks", default=None)
    parser.add_argument("--t", help="Print transition matrix to given"
                        " file in GRAPHVIZ DOT format.  Convert to PDF with "
                        " dot <file> -Tpdf > <outFile>", default=None)
    parser.add_argument("--minTP", help="Minimum tranisition probability "
                        "to include in transition matrix output from --t option.",
                        type=float, default=EPSILON)
    parser.add_argument("--minTPns", help="Minimum transition probability after "
                        "self transition is normalized out (ie after dividing by 1-self)",
                        type=float, default=EPSILON)
    parser.add_argument("--teStates", help="comma-separated list of state names"
                        " to consider TE-1, TE-2, ... etc", default=None)
    
    args = parser.parse_args()

    # load model created with teHmmTrain.py
    model = loadModel(args.inputModel)

    if args.teStates is not None:
        args.teStates = set(x for x in args.teStates.split(","))

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
        writeEmissionClusters(model, args, False)

    if args.ecn is not None:
        if canPlot is False:
            raise RuntimeError("Unable to write plots.  Maybe matplotlib is "
                               "not installed?")
        writeEmissionClusters(model, args, True)        

    if args.pca is not None:
        if canPlot is False:
            raise RuntimeError("Unable to write plots.  Maybe matplotlib is "
                               "not installed?")
        writeEmissionScatters(model, args)

    if args.hm is not None:
        if canPlot is False:
            raise RuntimeError("Unable to write plots.  Maybe matplotlib is "
                               "not installed?")
        writeEmissionHeatMap(model, args)

    if args.t is not None:
        writeTransitionGraph(model, args)


def writeEmissionClusters(model, args, onlyNonNumeric):
    """ print a hierachical clustering of states for each track (where each
    state is a point represented by its distribution for that track)"""
    trackList = model.getTrackList()
    stateNameMap = model.getStateNameMap()

    emission = model.getEmissionModel()
    # [TRACK][STATE][SYMBOL]
    emissionDist = np.exp(emission.getLogProbs())

    # leaf names of our clusters are the states
    if stateNameMap is not None:
        stateNames = map(stateNameMap.getMapBack, xrange(len(stateNameMap)))
    else:
        stateNames = [str(x) for x in xrange(model.n_components)]
    stateNames = applyTEStateNaming(args.teStates, stateNames)
    N = len(stateNames)

    # cluster for each track
    hcList = []
    hcNames = []
    # list for each state
    allPoints = []
    for i in xrange(N):
        allPoints.append([])

    for track in trackList:
        nonNumeric = False
        for symbol in emission.getTrackSymbols(track.getNumber()):
            try:
                val = float(track.getValueMap().getMapBack(symbol))
            except:
                nonNumeric = True
                break
        if nonNumeric is True or onlyNonNumeric is False:
            hcNames.append(track.getName())
            points = [emissionDist[track.getNumber()][x] for x in xrange(N)]
            for j in xrange(N):
                allPoints[j] += list(points[j])
            hc = hierarchicalCluster(points, normalizeDistances=True)
            hcList.append(hc)

    # all at once
    hc = hierarchicalCluster(allPoints, normalizeDistances=True)
    #hcList.append(hc)
    #hcNames.append("all_tracks")
    
    # write clusters to pdf (ranked in decreasing order based on total
    # branch length)
    ranks = rankHierarchies(hcList)
    outPath = args.ec
    if onlyNonNumeric is True:
        outPath = args.ecn
    plotHierarchicalClusters([hcList[i] for i in ranks],
                             [hcNames[i] for i in ranks],
                             stateNames, outPath)

def writeEmissionScatters(model, args):
    """ print a pca scatterplot of states for each track (where each
    state is a point represented by its distribution for that track)"""
    trackList = model.getTrackList()
    stateNameMap = model.getStateNameMap()
    emission = model.getEmissionModel()
    # [TRACK][STATE][SYMBOL]
    emissionDist = np.exp(emission.getLogProbs())

    # leaf names of our clusters are the states
    if stateNameMap is not None:
        stateNames = map(stateNameMap.getMapBack, xrange(len(stateNameMap)))
    else:
        stateNames = [str(x) for x in xrange(model.n_components)]
    stateNames = applyTEStateNaming(args.teStates, stateNames)

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

def writeEmissionHeatMap(model, args, leftTree = True, topTree = True):
    """ print a heatmap from the emission distributions.  this is of the form of
    track x state where each value is the man of the distribution for that state
    for that track"""
    trackList = model.getTrackList()
    stateNameMap = model.getStateNameMap()
    emission = model.getEmissionModel()
    # [TRACK][STATE][SYMBOL]
    emissionDist = np.exp(emission.getLogProbs())

    # leaf names of our clusters are the states
    if stateNameMap is not None:
        stateNames = map(stateNameMap.getMapBack, xrange(len(stateNameMap)))
    else:
        stateNames = [str(x) for x in xrange(model.n_components)]
    stateNames = applyTEStateNaming(args.teStates, stateNames)

    sortedStateNames = sorted(stateNames)
    stateRanks = [sortedStateNames.index(state) for state in stateNames]
    #stateNames = sortedStateNames

    N = len(stateNames)
    emProbs = emission.getLogProbs()

    # output means for each track
    # TRACK X STATE
    meanArray = []
    # keep track of track name for each row of meanArray
    trackNames = []

    # mean for each track
    for track in trackList:
        nonNumeric = False
        trackNo = track.getNumber()
        nameMap = track.getValueMap()
        trackMeans = np.zeros((emission.getNumStates()))

        for idx, state in enumerate(stateRanks):
            if track.getDist() == "gaussian":
                mean, stddev = emission.getGaussianParams(trackNo, state)
            else:
                mean = 0.0                
                for symbol in emission.getTrackSymbols(trackNo):
                    # do we need to check for missing value here???
                    val = nameMap.getMapBack(symbol)
                    temp = val
                    val = str(val)
                    if val != "None" and val != "0" and val.lower() != "simple" and \
                      val.lower() != "intergenic" and val.lower() != "low":
                        val = 1.
                    else:
                        val = 0.
    
                    prob = np.exp(emProbs[trackNo][state][symbol])

                    assert prob >= 0. and prob <= 1.
                    mean += val * prob                

            if nonNumeric is False:
                trackMeans[stateRanks[state]] = mean
            else:
                break                

        if nonNumeric is False:
            means = [trackMeans[state] for state in xrange(emission.getNumStates())]
            minVal = min(means)
            maxVal = max(means)
            for state in xrange(emission.getNumStates()):
                mean = trackMeans[state]
                # normalize mean
                if minVal != maxVal:
                    trackMeans[state] = (mean - minVal) / (maxVal - minVal)
                else:
                    trackMeans[state] = minVal
                #hacky cutoff
                #mean = min(0.23, mean)
        if nonNumeric is False:
            meanArray.append(trackMeans)
            trackNames.append(track.getName())

    # dumb transpose no time to fix ince
    tmeans = np.zeros((len(meanArray[0]), len(meanArray)))
    for i in xrange(len(meanArray)):
        for j in xrange(len(meanArray[i])):
            tmeans[j,i] = meanArray[i][j]
    #meanArray = tmeans

    if len(meanArray) > 0:
        # note to self http://stackoverflow.com/questions/2455761/reordering-matrix-elements-to-reflect-column-and-row-clustering-in-naiive-python
        plotHeatMap(meanArray, trackNames, sortedStateNames, args.hm, leftTree, topTree)
    

def writeTransitionGraph(model, args):
    """ write a graphviz text file """
    trackList = model.getTrackList()
    stateNameMap = model.getStateNameMap()
    if stateNameMap is not None:
        stateNames = map(stateNameMap.getMapBack, xrange(len(stateNameMap)))
    else:
        stateNames = [str(x) for x in xrange(model.n_components)]
    stateNames = applyTEStateNaming(args.teStates, stateNames)
    stateNames = map(lambda x: x.replace("-", "_"), stateNames)
    stateNames = map(lambda x: x.replace("|", "_"), stateNames)
    
    N = len(stateNames)
    f = open(args.t, "w")
    f.write("Digraph G {\n")
    for i, state in enumerate(stateNames):
        for j, toState in enumerate(stateNames):
            tp = model.getTransitionProbs()[i, j]
            tpNor = tp
            selfProb = model.getTransitionProbs()[i, i]
            if selfProb > EPSILON:
                tpNor = tp / (1. - selfProb)
            if tp > args.minTP and i != j and tpNor > args.minTPns:
                label = "label=\"%.2f\"" % (tp * 100.)
                width = "penwidth=%d" % (1 + int(tp / 20))
                f.write("%s -> %s [%s,%s];\n" % (state, toState, label, width))
    f.write("}\n")
    f.close()

def applyTEStateNaming(teStates, states):
    if teStates is None or len(teStates) is 0:
        return states
    
    teCount = 0
    otherCount = 0
    output = []
    for state in states:
        if state in teStates:
            output.append("TE-%d" % teCount)
            teCount += 1
        else:
            output.append("Other-%d" % otherCount)
            otherCount += 1

    if teCount + otherCount == 0:
        return states
    return output


if __name__ == "__main__":
    sys.exit(main())
