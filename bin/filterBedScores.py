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
Filter out bed intervals that are not within given score threshold (or rename
them)
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter out bed intervals that are outside [minScore, "
        "maxScore] (or rename them)")
    parser.add_argument("inputBed", help="Bed file to filter")
    parser.add_argument("minScore", help="Minimum score.  Intervals with score"
                        " < this amount will be filtered out.", type=float)
    parser.add_argument("maxScore", help="Minimum score.  Intervals with score"
                        " > this amount will be filtered out.", type=float)
    parser.add_argument("--rename", help="Instead of removing intervals that "
                        "are outside [minScore, maxScore], rename them",
                        default=None)
    parser.add_argument("--names", help="Comma-separated list of IDs ("
                        "0-based 3rd column) to apply filter to.  All elements"
                        " by default", default=None)
                        
    
    args = parser.parse_args()
    assert os.path.isfile(args.inputBed)
    nameSet = set()
    if args.names is not None:
        nameSet = set(args.names.split(","))
    
    tempBedToolPath = initBedTool()

    bedIntervals = BedTool(args.inputBed).sort()
    
    for interval in bedIntervals:
        doFilter = interval.name in nameSet and\
          (float(interval.score) < args.minScore or
           float(interval.score) > args.maxScore)
        if doFilter is False:
            sys.stdout.write("%s" % str(interval))
        elif args.rename is not None:
            interval.name = args.rename
            sys.stdout.write("%s" % str(interval))

    cleanBedTool(tempBedToolPath)
    
if __name__ == "__main__":
    sys.exit(main())
