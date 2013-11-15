#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging

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

    parser.add_argument("bed1", help="Bed file")
    parser.add_argument("bed2", help="Bed file covering same regions in same"
                        " order as bed1")
    parser.add_argument("--col", help="Column of bed files to use for state"
                        " (currently only support 4(name) or 5(score))",
                        default = 4, type = int)

    args = parser.parse_args()

    assert args.col == 3 or args.col ==4

    intervals1 = readBedIntervals(args.bed1, ncol = args.col)
    intervals2 = readBedIntervals(args.bed2, ncol = args.col)
    stats = compareIntervals(intervals1, intervals2, args.col)

def compareIntervals(intervals1, intervals2, col):
    """ return dictionary that maps each state to (i1 but not i2, i2 but not i1,
    both)"""

    assert intervals1[0][0] == intervals2[0][0]
    assert intervals1[0][1] == intervals2[0][1]
    assert intervals1[-1][2] == intervals2[-1][2]
 
    stats = dict()
    # yuck:
    p2 = 0
    for p1 in xrange(len(intervals1)):
        i1 = intervals1[p1]
        for pos in xrange(i1[2] - i1[1]):
            i2 = intervals2[p2]
            chrom = i1[0]
            coord = i1[1] + pos
            if i2[0] != chrom or not (coord >= i2[1] and coord < i2[2]):
                p2 += 1
                i2 = intervals[p2]
                assert i2[0] == chrom and coord >= i2[1] and cord < i2[2]
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
    
if __name__ == "__main__":
    sys.exit(main())
