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

from teHmm.trackIO import readBedIntervals
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

    args = parser.parse_args()

    assert args.col == 3 or args.col ==4

    intervals1 = readBedIntervals(args.bed1, ncol = args.col)
    intervals2 = readBedIntervals(args.bed2, ncol = args.col)
    stats = compareIntervals(intervals1, intervals2, args.col - 1)

    totalRight, totalWrong, accMap = summarizeComparision(stats)
    print stats
    totalBoth = totalRight + totalWrong
    accuracy = float(totalRight) / float(totalBoth)
    print "Accuaracy: %d / %d = %f" % (totalRight, totalBoth, accuracy)
    print "State-by-state (Precision, Recall):"
    print accMap

    # print some row data to be picked up by scrapeBenchmarkRow.py
    header, row = summaryRow(accuracy, stats, accMap)
    print " ".join(header)
    print " ".join(row)

    # make graph
    if args.plot is not None:
        if canPlot is False:
            raise RuntimeError("Unable to write plots.  Maybe matplotlib is "
                               "not installed?")
        writeAccPlots(accuracy, stats, accMap, intStats, intAccMap, args.plot)


def compareIntervals(intervals1, intervals2, col):
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
    
def summarizeComparision(stats):
    totalRight = 0
    totalWrong = 0
    accMap = dict()
    for state, stat in stats.items():
        totalRight += stat[2]
        totalWrong += stat[0] + stat[1]
        tp = float(stat[2])
        fn = float(stat[0])
        fp = float(stat[1])
        accMap[state] = (tp / (np.finfo(float).eps + tp + fp),
                         tp / (np.finfo(float).eps + tp + fn))
    return (totalRight, totalWrong, accMap)

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

def writeAccPlots(accuracy, stats, accMap, intStats, intAccMap, outFile):
    """ plot accuracies as scatter plots"""
    distList = [[], []]
    titles = []
    stateNames = []
    fscore = [[], []] 

    for state in sorted(accMap.keys()):
        acc = accMap[state]
        prec = acc[0]
        rec = acc[1]

        intAcc = intAccMap[state]
        intPrec = intAcc[0]
        intRec = intAcc[1]

        if (prec > 0.0 or rec > 0.0) and (intPrec > 0.0 or intRec > 0.0):
            stateNames.append(state)
            fs = 2 * ((prec * rec) / (prec + rec))
            ifs = 2 * ((intPrec * intRec) / (intPrec + intRec))
            fscore[0].append(fs)
            fscore[1].append(ifs)
            distList[0].append((prec, rec))
            distList[1].append((intPrec, intRec))

    titles.append("Base Acc. (avg f1score=%.3f)" % np.mean(fscore[0]))
    titles.append("Inteval Acc. (avg f1score=%.3f)" % np.mean(fscore[1]))
    plotPoints2d(distList[:1], titles, stateNames, outFile, xRange=(0,1.1),
                 yRange=(0, 1.4), ptSize=50, xLabel="Precision",
                 yLabel="Recall", cols=1, width=5, rowHeight=5)
            
if __name__ == "__main__":
    sys.exit(main())
