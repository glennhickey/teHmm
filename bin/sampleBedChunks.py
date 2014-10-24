#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import copy
import random

from teHmm.common import intersectSize, initBedTool, cleanBedTool, runShellCommand
from teHmm.trackIO import readBedIntervals

"""
Subselect some output of chunkBedRegions.py"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Subselect some output of chunkBedRegions.py")
    parser.add_argument("inBed", help="Input bed file (generated with chunkBedRegions.py)")
    parser.add_argument("sampleSize", help="Desired sample size (in bases).", type=int)
    
    args = parser.parse_args()
    tempBedToolPath = initBedTool()
    assert os.path.exists(args.inBed)

    bedIntervals = readBedIntervals(args.inBed)
    outIntervals = []

    curSize = 0

    # dumb n^2 alg should be enough for our current purposes
    while curSize < args.sampleSize and len(bedIntervals) > 0:
        idx = random.randint(0, len(bedIntervals)-1)
        interval = bedIntervals[idx]
        sampleLen = interval[2] - interval[1]
        if sampleLen + curSize > args.sampleSize:
            sampleLen = (sampleLen + curSize) - args.sampleSize
            interval = (interval[0], interval[1], interval[1] + sampleLen)
        outIntervals.append(interval)
        curSize += sampleLen
        del bedIntervals[idx]

    for interval in sorted(outIntervals):
        sys.stdout.write("%s\t%d\t%d\n" % interval)

    cleanBedTool(tempBedToolPath)
            
if __name__ == "__main__":
    sys.exit(main())
