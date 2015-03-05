#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import numpy as np
import copy

from teHmm.trackIO import readBedIntervals
from teHmm.modelIO import loadModel
from teHmm.common import intersectSize, initBedTool, cleanBedTool, runShellCommand
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.bin.compareBedStates import compareIntervalsOneSided, checkExactOverlap
from teHmm.bin.compareBedStates import compareBaseLevel, cutOutMaskIntervals
from teHmm.bin.compareBedStates import getStateMapFromConfMatrix, getStateMapFromConfMatrix_simple

try:
    from teHmm.parameterAnalysis import plotHeatMap
    canPlot = True
except:
    canPlot = False


""" Given two bed files: a prediction and a true (or target) annotation,
    re-label the prediction's state names so that they best match the
    true annotation.  Usees same logic as compareBedStates.py for
    determining accuracy """

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=" Given two bed files: a prediction and a true (or target)"
         " annotation, re-label the prediction's state names so that they "
         " best match the true annotation.  Usees same logic as "
         " compareBedStates.py for determining accuracy")

    parser.add_argument("tgtBed", help="Target bed file")
    parser.add_argument("predBed", help="Predicted bed file to re-label. ")
    parser.add_argument("outBed", help="Output bed (relabeling of predBed)")
    parser.add_argument("--col", help="Column of bed files to use for state"
                        " (currently only support 4(name) or 5(score))",
                        default = 4, type = int)
    parser.add_argument("--intThresh", help="Threshold to consider interval from"
                        " tgtBed covered by predBed.  If not specified, then base"
                        " level statistics will be used. Value in range (0,1]",
                        type=float, default=None)
    parser.add_argument("--noFrag", help="Dont allow fragmented interval matches ("
                        "see help for --frag in compareBedStates.py).  Only"
                        " relevant with --intThresh", action="store_true",
                        default=False)
    parser.add_argument("--qualThresh", help="Minimum match ratio between truth"
                        " and prediction to relabel prediction.  Example, if"
                        " predicted state X overlaps target state LTR 25 pct of "
                        "the time, then qualThresh must be at least 0.25 to "
                        "label X as LTR in the output.  Value in range (0, 1]",
                        type=float, default=0.1)
    parser.add_argument("--ignore", help="Comma-separated list of stateNames to"
                        " ignore (in prediction)", default=None)
    parser.add_argument("--ignoreTgt", help="Comma-separated list of stateNames to"
                        " ignore (int target)", default=None)
    parser.add_argument("--tgt", help="Comma-separated list of stateNames to "
                        " consider (in target).  All others will be ignored",
                        default=None)
    parser.add_argument("--unique", help="If more than one predicted state maps"
                        " to the same target state, add a unique id (numeric "
                        "suffix) to the output so that they can be distinguished",
                        action="store_true", default=False)
    parser.add_argument("--model", help="Apply state name mapping to the model"
                        " in the specified path (it is strongly advised to"
                        " make a backup of the model first)", default=None)
    parser.add_argument("--noMerge", help="By default, adjacent intervals"
                        " with the same state name in the output are "
                        "automatically merged into a single interval.  This"
                        " flag disables this.", action="store_true",
                        default=False)
    parser.add_argument("--hm", help="Write confusion matrix as heatmap in PDF"
                        " format to specified file", default = None)
    parser.add_argument("--old", help="Use old name mapping logic which just "
                        "takes biggest overlap in forward confusion matrix.  "
                        "faster than new default logic which does the greedy"
                        " f1 optimization", action="store_true", default=False)
    parser.add_argument("--fdr", help="Use FDR cutoff instead of (default)"
                        " greedy F1 optimization for state labeling",
                        type=float, default=None)
    parser.add_argument("--tl", help="Path to tracks XML file.  Used to cut "
                        "out mask tracks so they are removed from comparison."
                        " (convenience option to not have to manually run "
                        "subtractBed everytime...)", default=None)
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    if args.ignore is not None:
        args.ignore = set(args.ignore.split(","))
    else:
        args.ignore = set()
    if args.ignoreTgt is not None:
        args.ignoreTgt = set(args.ignoreTgt.split(","))
    else:
        args.ignoreTgt = set()
    if args.tgt is not None:
        args.tgt = set(args.tgt.split(","))
        if args.old is True:
            raise RuntimeError("--tgt option not implemented for --old")
    else:
        args.tgt = set()
    if args.old is True and args.fdr is not None:
        raise RuntimeError("--old and --fdr options are exclusive")

    assert args.col == 4 or args.col == 5

    tempFiles = []
    if args.tl is not None:
        cutBedTgt = cutOutMaskIntervals(args.tgtBed, -1, sys.maxint, args.tl)                                
        cutBedPred = cutOutMaskIntervals(args.predBed, -1, sys.maxint, args.tl)
        
        if cutBedTgt is not None:
            assert cutBedPred is not None
            tempFiles += [cutBedTgt, cutBedPred]
            args.tgtBed = cutBedTgt
            args.predBed = cutBedPred

    checkExactOverlap(args.tgtBed, args.predBed)

    intervals1 = readBedIntervals(args.tgtBed, ncol = args.col)
    intervals2 = readBedIntervals(args.predBed, ncol = args.col)
    cfName = "reverse"

    if args.old is True:
        intervals1, intervals2 = intervals2, intervals1
        cfName = "forward"

    # generate confusion matrix based on accuracy comparison using
    # base or interval stats as desired
    if args.intThresh is not None:
        logger.info("Computing interval %s confusion matrix" % cfName)
        confMat = compareIntervalsOneSided(intervals2, intervals1, args.col -1,
                                            args.intThresh, False,
                                           not args.noFrag)[1]
    else:
        logger.info("Computing base %s confusion matrix" % cfName)
        confMat = compareBaseLevel(intervals2, intervals1, args.col - 1)[1]

    logger.info("%s Confusion Matrix:\n%s" % (cfName, str(confMat)))

    # find the best "true" match for each predicted state    
    if args.old is True:
        intervals1, intervals2 = intervals2, intervals1
        stateMap = getStateMapFromConfMatrix_simple(confMat)
    else:
        stateMap = getStateMapFromConfMatrix(confMat, args.tgt, args.ignoreTgt,
                                             args.ignore, args.qualThresh,
                                             args.fdr)

    # filter the stateMap to take into account the command-line options
    # notably --ignore, --ignoreTgt, --qualThresh, and --unique
    filterStateMap(stateMap, args)

    logger.info("State Map:\n%s", str(stateMap))
        
    # write the model if spefied
    if args.model is not None:
        applyNamesToModel(stateMap, args.model)
    
    # generate the output bed using the statemap
    writeFittedBed(intervals2, stateMap, args.outBed, args.col-1, args.noMerge,
                   args.ignoreTgt)

    # write the confusiont matrix as heatmap
    if args.hm is not None:
        if canPlot is False:
            raise RuntimeError("Unable to write heatmap.  Maybe matplotlib is "
                               "not installed?")
        writeHeatMap(confMat, args.hm)

    if len(tempFiles) > 0:
        runShellCommand("rm -f %s" % " ".join(tempFiles))
    cleanBedTool(tempBedToolPath)

def filterStateMap(stateMap, args):
    """ Make sure ignored states are ignored.  Apply unique id suffixes is necessary.
    Make sure that quality threshold is high enough or else ignore too.  map is
    filtered in place."""
    mapCounts = dict()
    for name, mapVal in stateMap.items():
        assert len(mapVal) == 3
        mapName, mapCount, mapTotal = mapVal
        qual = float(mapCount) / float(mapTotal)
        if name in args.ignore or qual < args.qualThresh:
            # set map such that it won't be changed
            logger.debug("Ignoring state %s with quality %f" % (name, qual))
            stateMap[name] = (name, 1, 1)
        elif args.unique:
            if mapName not in mapCounts:
                mapCounts[mapName] = 1
            else:
                mapCounts[mapName] += 1

    # dont want to rename if only 1 instance
    for name, count in mapCounts.items():
        if count == 1:
            mapCounts[name] = 0

    # 2nd pass to assign the unique ids (range from 1 to count)
    for name, mapVal in stateMap.items():
        mapName, mapCount, mapTotal = mapVal
        if mapName in mapCounts:
            count = mapCounts[mapName]
            if count > 0:
                newName = mapName + ".%d" % count
                logger.debug("Mapping %s to %s" % (mapName, newName))
                stateMap[name] = (newName, mapCount, mapTotal)
                mapCounts[mapName] -= 1

def applyNamesToModel(stateMap, modelPath):
    """ change a given HMM model to use the new state names"""
    # load model created with teHmmTrain.py
    logger.debug("loading model %s" % modelPath)
    model = loadModel(modelPath)
    modelMap = model.getStateNameMap()
    raise RuntimeError("Not Implemented")
                
def writeFittedBed(intervals, stateMap, outBed, col, noMerge, ignoreTgt):
    """ write the mapped bed file by applying stateMap to the intervals
    from args.predBed"""
    outFile = open(outBed, "w")

    prevInterval = None
    for interval in intervals:
        outInterval = list(interval)
        if outInterval[col] in stateMap and\
          stateMap[outInterval[col]][0] not in ignoreTgt:
            outInterval[col] = stateMap[outInterval[col]][0]
        if not noMerge and\
          prevInterval is not None and\
          outInterval[col] == prevInterval[col] and\
          outInterval[0] == prevInterval[0] and\
          outInterval[1] == prevInterval[2]:
            # glue onto prev interval
            prevInterval[2] = outInterval[2]
        else:
            # write and update prev
            if prevInterval is not None:
                outFile.write("\t".join([str(x) for x in prevInterval]) + "\n")
            prevInterval = outInterval
            
    if prevInterval is not None:
        outFile.write("\t".join([str(x) for x in prevInterval]) + "\n")
                                            
    outFile.close()

def writeHeatMap(confMat, outPath):
    """ make a heatmap PDF out of a confusion matrix using """

    # need to transform our dict[dict] confusion matrix into an array
    fromStates = set()
    toStates = set()
    fromTotals = dict()
    for fromState, fsDict in confMat.items():
        fromStates.add(fromState)
        if fromState not in fromTotals:
            fromTotals[fromState] = 0
        for toState, count in fsDict.items():
            toStates.add(toState)
            fromTotals[fromState] += count
    fromStates = list(fromStates)
    toStates = list(toStates)
    toRanks = [sorted(toStates).index(i) for i in toStates]
    frRanks = [sorted(fromStates).index(i) for i in fromStates]

    matrix = np.zeros((len(fromStates), len(toStates)))
    for fromIdx in xrange(len(fromStates)):
        for toIdx in xrange(len(toStates)):
            fromState = fromStates[fromIdx]
            toState = toStates[toIdx]
            count = 0.
            # dumb we need if -- should add zero-entries into confMat
            # (or at least change to defaultDict...)
            if fromState in confMat and toState in confMat[fromState]:
                count = float(confMat[fromState][toState])
                # normalize
                count /= float(fromTotals[fromState])
            matrix[frRanks[fromIdx], toRanks[toIdx]] = count

    plotHeatMap(matrix.T, sorted(toStates), sorted(fromStates), outPath)

if __name__ == "__main__":
    sys.exit(main())
