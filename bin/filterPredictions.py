#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import copy

from pybedtools import BedTool, Interval
from teHmm.common import initBedTool, cleanBedTool

"""
Really simple script to clean out some obviously spurious HMM predictions by
renaming them to the outside state.  More care wrt to parameters logic
will probably be needed down the road, since we hardcode most everything for now
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter out some obviously spurious predictions. Will"
        " probably be need to be called many times to be useful.  Order "
        " of precedence of filters is length, mustBefore, mustAfter,"
        "mergeBefore, mergeAfter")
    parser.add_argument("inputBed", help="Bed file to filter")
    parser.add_argument("--outside", help="Name of outside state to rename"
                        " filtered elements to", default="Outside")
    parser.add_argument("--minLen", help="Filter lenTargets that are smaller"
                        " than this length", default=0, type=int)
    parser.add_argument("--lenTgts", help="Comma-separated list of state names"
                        " to apply filter to or * for all.  Make sure to use"
                        " \"s for special characters including *", default=None)
    parser.add_argument("--mustBefore", help="Comma-separated list of states"
                        " where the first one will be filtered out if it does"
                        " not preceed any of the other states. "
                        "Ex: \"LTR|left,inside,DNA\" "
                        " would filter out LTR|left if not before an inside"
                        " or DNA element",
                        default=None)
    parser.add_argument("--mustAfter", help="Comma-separated list of states"
                        " where the first one will be filtered out if it does"
                        " not follow any of the other states. "
                        "  Ex: \"LTR|right,inside,DNA\" "
                        " would filter out LTR|right if not after an insde or"
                        " DNA element",
                        default=None)
    parser.add_argument("--mergeBefore", help="Comma-separated list of states"
                        " where the first one will be merged with its right "
                        "neighbour (adopting neighbours name) if it preceeds"
                        " any of the other states.",
                        default=None)
    parser.add_argument("--mergeAfter", help="Comma-separated list of states"
                        " where the first one will be merged with its left "
                        "neighbour (adopting neighbours name) if it follows"
                        " any of the other states.",
                        default=None)
    parser.add_argument("--valCol", help="0-based column to look for state "
                        "name", type = int, default = 3)

    args = parser.parse_args()
    tempBedToolPath = initBedTool()
    assert os.path.isfile(args.inputBed)
    assert args.valCol == 3 or args.valCol == 4

    lenTgtSet = set()
    if args.lenTgts is not None:
        lenTgtSet = set(args.lenTgts.split(","))

    mustBefore = None
    if args.mustBefore is not None:
        mustBefore = args.mustBefore.split(",")
        assert len(mustBefore) > 1

    mustAfter = None
    if args.mustAfter is not None:
        mustAfter = args.mustAfter.split(",")
        assert len(mustAfter) > 1

    mergeBefore = None
    if args.mergeBefore is not None:
        mergeBefore = args.mergeBefore.split(",")
        assert len(mergeBefore) > 1

    mergeAfter = None
    if args.mergeAfter is not None:
        mergeAfter = args.mergeAfter.split(",")
        assert len(mergeAfter) > 1
        
    bedIntervals = [i for i in BedTool(args.inputBed).sort()]

    def getName(interval):
        if args.valCol == 3:
            return interval.name
        if args.valCol == 4:
            return interval.score
        return None

    outIntervals = []
    for i in xrange(len(bedIntervals)):
        prevInterval = None
        if i > 0:
            prevInterval = bedIntervals[i-1]
        nextInterval = None
        if i < len(bedIntervals) - 1:
            nextInterval = bedIntervals[i+1]
        curInterval = bedIntervals[i]
        curName = getName(curInterval)
        filter = False

        # filter by length
        if (curName in lenTgtSet and
            curInterval.end - curInterval.start < args.minLen):
            filter = True

        # filter by mustBefore
        if (filter is False and
            mustBefore is not None and
            curName == mustBefore[0] and
            nextInterval is not None and
            getName(nextInterval) not in mustBefore[1:]):
            filter = True

        # filter by mustAfter
        if (filter is False and
            mustAfter is not None and
            curName == mustAfter[0] and
            prevInterval is not None and
            getName(prevInterval) not in mustAfter[1:]):
            filter = True

        if filter is True:
            curInterval = copy.deepcopy(curInterval)
            curInterval.name = args.outside

        merge = False
        # mergeBefore (current is absorbed into next)
        if (filter is False and
            mergeBefore is not None and
            curName == mergeBefore[0] and
            nextInterval is not None and
            getName(nextInterval) in mergeBefore[1:]):
            nextInterval.start = curInterval.start
            merge = True

        # mergeAfter (current is absorbed into next. next takes current name)
        if (filter is False and merge is False and
            mergeAfter is not None and
            nextInterval is not None and
            getName(nextInterval) == mergeAfter[0] and
            curName in mergeAfter[1:]):
            nextInterval.start = curInterval.start
            nextInterval.name = curName
            merge = True

        if merge is False:
            outIntervals.append(curInterval)
            

    for outInterval in outIntervals:
        sys.stdout.write(str(outInterval))

    cleanBedTool(tempBedToolPath)
    
if __name__ == "__main__":
    sys.exit(main())
