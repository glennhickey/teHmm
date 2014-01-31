#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import copy

from pybedtools import BedTool, Interval

"""
Script to chop up bed intervals in order to make more states out of them.  To
be used to, say, transform some 5-state LTR_FINDER supervised training data
into 15-state data.

For example, given

LTR|left   10  20

The output could be (if chopping 2 on left and 3 on right with size 1)

LTR|left_l2   10   11
LTR|left_l1   11   12
LTR|left      12   16
LTR|left_r1   17   18
LTR|left_r2   18   19
LTR|left_r3   19   20

Note that the output slices for any given input line will be sorted.  If the
input lines are overlapping or out of order than the output will be too.

"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Chop off ends of target bed intervals and give "
        "pieces new names.")
    parser.add_argument("inBed", help="Input bed file")
    parser.add_argument("paramsFile", help="Text file containing some chop"
                        " parameters. In particular, each line must have"
                        " at 5 columns: <name> <leftSliceSize> <numLeftSlices> "
                        " <rightSliceSize> <numRightSlices> to configure the "
                        "slicing for a given state.")
    parser.add_argument("outBed", help="bed to write output to.")
    parser.add_argument("--leftSuffix", help="suffix for new left slice names",
                        default="_l")
    parser.add_argument("--rightSuffix", help="suffix for new right slice "
                        "names", default="_r")
                        
    
    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    outFile = open(args.outBed, "w")

    cutParams = parseCutFile(args.paramsFile)
    
    for interval in BedTool(args.inBed):
        newIntervals = cutInterval(interval, cutParams, args)
        for newInterval in newIntervals:
            outFile.write(str(newInterval))

    outFile.close()


def parseCutFile(path):
    """ parse the parameters file.  white-space separated columns of form
    <name> <leftSliceSize> <numLeftSlices> <rightSliceSize> <numRightSlices>
    Returns a dictionary mapping <name> to a 4-tuple representing the remaining
    4 columns"""
    cutParams = dict()
    pfile = open(path, "r")
    for line in pfile:
        lsline = line.lstrip()
        if len(lsline) > 0 and lsline[0] != "#":
            toks = lsline.split()
            assert len(toks) == 5
            assert toks[0] not in cutParams
            cutParams[toks[0]] = (int(toks[1]), int(toks[2]), int(toks[3]),
                                  int(toks[4]))
    return cutParams

def cutInterval(interval, cutParams, args):
    """ do the cutting if the interval is in the cutparams, otherwise just
    return it untouched.  when making new intervals, the name and coordinates
    get changed, but that's it (all other columns untouched). """
    if interval.name not in cutParams:
        return [interval]
    leftIntervals, rightIntervals = [], []
    leftIdx, rightIdx = 0, 0
    ilen = interval.end - interval.start
    
    leftSize, numLeft, rightSize, numRight = cutParams[interval.name]
    mSize = min(leftSize, rightSize)

    # keep cutting a slice off each side while we can still leave at least
    # one base of input
    while True:
        added = False
        if ilen > leftSize and leftIdx < numLeft:
            added = True
            ilen -= leftSize
            leftIdx += 1
            newInt = copy.deepcopy(interval)
            newInt.name = interval.name + args.leftSuffix + str(leftIdx)
            newInt.end = newInt.start + leftIdx * leftSize
            newInt.start = newInt.end - leftSize
            leftIntervals.append(newInt)
        
        if ilen > rightSize and rightIdx < numRight:
            added = True
            ilen -= rightSize
            rightIdx += 1
            newInt = copy.deepcopy(interval)
            newInt.name = interval.name + args.rightSuffix + str(rightIdx)
            newInt.start = newInt.end - rightIdx * rightSize
            newInt.end = newInt.start + rightSize
            rightIntervals.append(newInt)

        if added is False:
            break

    # edit input interval in place
    interval.start += leftIdx * leftSize
    interval.end -= rightIdx * rightSize
    assert interval.start < interval.end
    leftIntervals.append(interval)

    return leftIntervals + rightIntervals
            
            
if __name__ == "__main__":
    sys.exit(main())
