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
import copy

from teHmm.trackIO import readBedIntervals
from teHmm.common import intersectSize

try:
    from teHmm.parameterAnalysis import pcaFlatten, plotPoints2d
    canPlot = True
except:
    canPlot = False

""" Compare bed files (EX Truth vs. Viterbi output).  They must cover same
genomic region in the same order """

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare two bed files where Model states are represented"
        " in a column.  Used to determine sensitivity and specificity.  NOTE"
        " that both bed files must be sorted and cover the exact same regions"
        " of the same genome.")

    parser.add_argument("bed1", help="Bed file (TRUTH)")
    parser.add_argument("bed2", help="Bed file covering same regions in same"
                        " order as bed1")
    parser.add_argument("--col", help="Column of bed files to use for state"
                        " (currently only support 4(name) or 5(score))",
                        default = 4, type = int)
    parser.add_argument("--thresh", help="Threshold to consider interval from"
                        " bed1 covered by bed2.  NOTE: this analysis does not"
                        " presently care how many individual intervals cover "
                        "a given interval.  IE 10 bed lines of length 1 in bed1"
                        " that overlap a single line of length 10 in bed2 will"
                        "still be considered a perfect match.",
                        type=float, default=0.8)
    parser.add_argument("--plot", help="Path of file to write Precision/Recall"
                        " graphs to in PDF format", default=None)
    parser.add_argument("--ignore", help="Comma-separated list of stateNames to"
                        " ignore", default=None)

    args = parser.parse_args()
    if args.ignore is not None:
        args.ignore = set(args.ignore.split(","))
    else:
        args.ignore = set()

    assert args.col == 3 or args.col ==4

    intervals1 = readBedIntervals(args.bed1, ncol = args.col)
    intervals2 = readBedIntervals(args.bed2, ncol = args.col)
    stats = compareBaseLevel(intervals1, intervals2, args.col - 1)

    totalRight, totalWrong, accMap = summarizeBaseComparision(stats, args.ignore)
    print stats
    totalBoth = totalRight + totalWrong
    accuracy = float(totalRight) / float(totalBoth)
    print "Accuaracy: %d / %d = %f" % (totalRight, totalBoth, accuracy)
    print "State-by-state (Precision, Recall):"
    print accMap

    trueStats = compareIntervalsOneSided(intervals1, intervals2, args.col -1,
                                         args.thresh)
    predStats = compareIntervalsOneSided(intervals2, intervals1, args.col -1,
                                         args.thresh)
    intAccMap = summarizeIntervalComparison(trueStats, predStats, False,
                                            args.ignore)
    intAccMapWeighted = summarizeIntervalComparison(trueStats, predStats, True,
                                                     args.ignore)
    print "\nInterval Accuracy"
    print intAccMap
    print ""

    print "\nWeighted Interval Accuracy"
    print intAccMapWeighted
    print ""


    # print some row data to be picked up by scrapeBenchmarkRow.py
    header, row = summaryRow(accuracy, stats, accMap)
    print " ".join(header)
    print " ".join(row)

    # make graph
    if args.plot is not None:
        if canPlot is False:
            raise RuntimeError("Unable to write plots.  Maybe matplotlib is "
                               "not installed?")
        writeAccPlots(accuracy, accMap, intAccMap, intAccMapWeighted,
                      args.thresh, args.plot)


def compareBaseLevel(intervals1, intervals2, col):
    """ return dictionary that maps each state to (i1 but not i2, i2 but not i1,
    both) for base level stats, and also a similar dictionary for interval
    stats. """

    assert intervals1[0][0] == intervals2[0][0]
    assert intervals1[0][1] == intervals2[0][1]
    assert intervals1[-1][2] == intervals2[-1][2]

    # base level dictionary
    stats = dict()
    # yuck:
    p2 = 0
    for p1 in xrange(len(intervals1)):
        i1 = intervals1[p1]
        assert len(i1) > col
        for pos in xrange(i1[2] - i1[1]):
            i2 = intervals2[p2]
            assert len(i2) > col
            chrom = i1[0]
            coord = i1[1] + pos
            if i2[0] != chrom or not (coord >= i2[1] and coord < i2[2]):
                p2 += 1
                i2back = i2
                i2 = intervals2[p2]
                assert i2[0] == chrom and coord >= i2[1] and coord < i2[2]
            state1 = i1[col]
            state2 = i2[col]
            if state1 not in stats:
                stats[state1] = [0, 0, 0]
            if state2 not in stats:
                stats[state2] = [0, 0, 0]
            if state1 == state2:
                stats[state1][2] += 1
            else:
                stats[state1][0] += 1
                stats[state2][1] += 1

    return stats

def compareIntervalsOneSided(trueIntervals, predIntervals, col, threshold):
    """ Same idea as baselevel comparison above, but treats bed intervals
    as single unit, and does not perform symmetric test.  In particular, we
    return the following stats here: for each true interval, is it covered
    by a predicted interval (with the same name) by at least threshold pct?
    The stats returned is therefore a pair for each state:
    (num intervals in truth correctly predicted , num intervals in truth
    incorrectly predicted)
    This is effectively a recall measure.  Of course, calling a second time
    with truth and pred swapped, will yield the precision.

    We also include the total lengths of the true predicted and false predicted
    elements.  So each states maps to a tuplie like
    (numTrue, totTrueLen, numFalse, totFalseLen)

    NOTE: this test will return a positive hit if a giant predicted interval
    overlaps a tiny true interval.  this can be changed, but since this form
    of innacuracy will be caught when called with true/pred swapped (precision)
    I'm not sure if it's necessary
    """

    # as in base level comp, both interval sets must cover exactly same regions
    # in same order.  the asserts below only partially check this:
    assert trueIntervals[0][0] == predIntervals[0][0]
    assert trueIntervals[0][1] == predIntervals[0][1]
    assert trueIntervals[-1][2] == predIntervals[-1][2]

    LP = len(predIntervals)
    LT = len(trueIntervals)

    stats = dict()
    
    pi = 0
    for ti in xrange(LT):

        trueInterval = trueIntervals[ti]
        trueState = trueInterval[col]
        trueLen = float(trueInterval[2] - trueInterval[1])
        
        # advance pi to first pred interval that intersects ti
        while True:
            if pi < LP and intersectSize(trueInterval,
                                         predIntervals[pi]) == 0:
                pi += 1
            else:
                break

        # scan all intersecting predIntervals with ti

        bestOverlap = 0
        for i in xrange(pi, LP):
            overlapSize = intersectSize(trueInterval, predIntervals[i])
            if overlapSize > 0:
                if predIntervals[i][col] == trueState:
                    bestOverlap = max(bestOverlap, overlapSize)
            else:
                break

        # update stats
        if trueState not in stats:
            stats[trueState] = [0, 0, 0, 0]

        if float(bestOverlap) / trueLen >= threshold:
            stats[trueState][0] += 1
            stats[trueState][1] += trueLen
        else:
            # dont really need this (can be inferred from total number of
            # true intervals but whatever)
            stats[trueState][2] += 1
            stats[trueState][3] += trueLen

    return stats
    
def summarizeBaseComparision(stats, ignore):
    totalRight = 0
    totalWrong = 0
    accMap = dict()
    for state, stat in stats.items():
        if state in ignore:
            continue
        totalRight += stat[2]
        totalWrong += stat[0] + stat[1]
        tp = float(stat[2])
        fn = float(stat[0])
        fp = float(stat[1])
        accMap[state] = (tp / (np.finfo(float).eps + tp + fp),
                         tp / (np.finfo(float).eps + tp + fn))
    return (totalRight, totalWrong, accMap)

def summarizeIntervalComparison(trueStats, predStats, weighted, ignore):
    """ like above but done on two 1-sided interval comparisions.  only
    retunrs a map (ie no totalright total wrong) """
    accMap = dict()
    stateSet = set(predStats.keys()).union(set(trueStats.keys())) - ignore

    totalTrueTp = 0
    totalTrueFp = 0
    totalPredTp = 0
    totalPredFp = 0
    
    for state in stateSet:
        recall = 0.0
        if state in trueStats:
            tp = trueStats[state][0]
            fp = trueStats[state][2]
            if weighted is True:
                tp *= trueStats[state][1]
                fp *= trueStats[state][3]
            totalTrueTp += tp
            totalTrueFp += fp
            if tp + fp > 0:
                recall = float(tp) / float(tp + fp)

        precision = 0.0
        if state in predStats:
            tp = predStats[state][0]
            fp = predStats[state][2]
            if weighted is True:
                tp *= predStats[state][1]
                fp *= predStats[state][3]
            totalPredTp += tp
            totalPredFp += fp
            if tp + fp > 0:
                precision = float(tp) / float(tp + fp)

        accMap[state] = (precision, recall)

    totalRecall = 0.
    if totalTrueTp + totalTrueFp > 0:
        totalRecall = float(totalTrueTp) / float(totalTrueTp + totalTrueFp)
    totalPrecision = 0.
    if totalPredTp + totalPredFp > 0:
        totalPrecision = float(totalPredTp) / float(totalPredTp + totalPredFp)

    assert "Overall" not in accMap
    accMap["Overall"] = (totalPrecision, totalRecall)
    
    return accMap
        

def summaryRow(accuracy, stats, accMap):
    header = []
    row = []
    header.append("totAcc")
    row.append(accuracy)
    for state in sorted(accMap.keys()):
        acc = accMap[state]
        # precision
        header.append("%s_Prec" % state)
        row.append(acc[0])
        # recall
        header.append("%s_Rec" % state)
        row.append(acc[1])
        # fscore
        header.append("%s_F1" % state) 
        fscore = 0
        if (acc[0] > 0 and acc[1] > 0):
            fscore = 2 * ((acc[0] * acc[1]) / (acc[1] + acc[0]))
        row.append(fscore)
    row = map(str, row)
    assert len(header) == len(row)
    return header, row

def writeAccPlots(accuracy, baseAccMap, intAccMap, intAccMapWeighted,
                  threshold, outFile):
    """ plot accuracies as scatter plots"""

    accMaps = [baseAccMap, intAccMap, intAccMapWeighted]
    names = ["Base", "Interval thresh=%.2f" % threshold,
             "Weighted Interval thresh=%.2f" % threshold]

    stateNames = set()
    for am in accMaps:
        stateNames = stateNames.union(set(am.keys()))
    emptyStates = set()
    for state in stateNames:
        total = 0.
        for am in accMaps:
            if state in am:
                total += am[state][0]
                total += am[state][1]
        if total == 0.:
            emptyStates.add(state)
        
    stateNames = list(stateNames - emptyStates)

    distList = []
    for i in xrange(len(accMaps)):
        distList.append([(0,0)] * len(stateNames))
    titles = []

    for i, accMap in enumerate(accMaps):
        totalF = 0.0
        numF = 0.0
        for state in sorted(accMap.keys()):
            acc = accMap[state]
            prec = acc[0]
            rec = acc[1]

            if prec > 0.0 or rec > 0.0:
                stateIdx = stateNames.index(state)
                fs = 2. * ((prec * rec) / (prec + rec))
                totalF += fs
                numF += 1.
                distList[i][stateIdx] = (prec, rec)
            
        avgF = 0.
        if totalF > 0:
            avgF = totalF / numF
        titles.append("%s Acc. (avg f1=%.3f)" % (names[i], avgF))
    plotPoints2d(distList, titles, stateNames, outFile, xRange=(0,1.1),
                 yRange=(0, 1.4), ptSize=75, xLabel="Precision",
                 yLabel="Recall", cols=2, width=10, rowHeight=5)
            
if __name__ == "__main__":
    sys.exit(main())
