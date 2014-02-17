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

    args = parser.parse_args()

    assert args.col == 3 or args.col ==4

    intervals1 = readBedIntervals(args.bed1, ncol = args.col)
    intervals2 = readBedIntervals(args.bed2, ncol = args.col)
    stats, intStats = compareIntervals(intervals1, intervals2, args.col - 1,
                                       args.thresh)

    totalRight, totalWrong, accMap = summarizeComparision(stats)
    print stats
    totalBoth = totalRight + totalWrong
    accuracy = float(totalRight) / float(totalBoth)
    print "Accuaracy: %d / %d = %f" % (totalRight, totalBoth, accuracy)
    print "State-by-state (Precision, Recall):"
    print accMap

    totalIntRight, totalIntWrong, intAccMap = summarizeComparision(intStats)
    print "\nInterval-level stats (threshold=%f)" % args.thresh
    print intStats
    totalIntBoth = totalIntRight + totalIntWrong
    if totalIntBoth > 0:
        intAccuracy = float(totalIntRight) / float(totalIntBoth)
    else:
        intAccuracy = 0
    print "Accuaracy: %d / %d = %f" % (totalIntRight, totalIntBoth, intAccuracy)
    print "State-by-state (Precision, Recall):"
    print intAccMap
    print ""

    # print some row data to be picked up by scrapeBenchmarkRow.py
    header, row = summaryRow(accuracy, stats, accMap)
    print " ".join(header)
    print " ".join(row)

def compareIntervals(intervals1, intervals2, col, threshold):
    """ return dictionary that maps each state to (i1 but not i2, i2 but not i1,
    both) for base level stats, and also a similar dictionary for interval
    stats. """

    assert intervals1[0][0] == intervals2[0][0]
    assert intervals1[0][1] == intervals2[0][1]
    assert intervals1[-1][2] == intervals2[-1][2]

    # base level dictionary
    stats = dict()
    # interval level dictionary
    intStats = dict()
    runningTally1 = [0, 0] #right / wrong
    runningTally2 = [0, 0] #right / wrong
    # yuck:
    p2 = 0
    i1 = [None] * (col + 1)
    for p1 in xrange(len(intervals1)):
        updateIntStats(i1[col], runningTally1, intStats, 1, threshold)
        i1 = intervals1[p1]
        assert len(i1) > col
        for pos in xrange(i1[2] - i1[1]):
            i2 = intervals2[p2]
            assert len(i2) > col
            chrom = i1[0]
            coord = i1[1] + pos
            if i2[0] != chrom or not (coord >= i2[1] and coord < i2[2]):
                updateIntStats(i2[col], runningTally2, intStats, 2, threshold)
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
                runningTally1[0] += 1
                runningTally2[0] += 1
            else:
                stats[state1][0] += 1
                stats[state2][1] += 1
                runningTally1[1] += 1
                runningTally2[1] += 1

    return stats, intStats

def updateIntStats(state, tally, intStats, idx, threshold):
    """ update interval-level stats using the threshold.  this function
    is called when changing an interval in either file (passing the appropraite
    tally).  The stats collected are the same as the base level stats:
    trio of form (i1 but not i2, i2 but not i1, both) but instead of bases
    we count entire intevarval (using threshold)"""
    assert len(tally) == 2
    if tally != [0, 0]:
        if state not in intStats:
            intStats[state] = [0, 0, 0]
        frac = float(tally[0]) / float(tally[0] + tally[1])
        if frac >= threshold:
            intStats[state][2] += 1
        elif idx == 1:
            intStats[state][0] += 1
        else:
            assert idx == 2
            intStats[state][1] += 1

    tally[0] = 0
    tally[1] = 0
    
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
    

if __name__ == "__main__":
    sys.exit(main())
