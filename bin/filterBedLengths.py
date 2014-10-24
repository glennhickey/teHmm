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
Filter out bed intervals that are smaller than a cutoff
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter out bed intervals with lengths outside given range")
    parser.add_argument("inputBed", help="Bed file to filter")
    parser.add_argument("minLength", help="Minimum interval length (inclusive)", type=int)
    parser.add_argument("maxLength", help="Maximum interval length (inclusive)", type=int)
    parser.add_argument("--rename", help="Rename id column to given name instead "
                        "of removing", default=None)
    parser.add_argument("--inv", help="Filter out lengths that are *inside* the given range",
                        action="store_true", default=False)
    
    args = parser.parse_args()
    assert os.path.isfile(args.inputBed)
    
    tempBedToolPath = initBedTool()

    bedIntervals = BedTool(args.inputBed).sort()
    
    for interval in bedIntervals:
        intervalLen = interval.end - interval.start
        filter = intervalLen < args.minLength or intervalLen > args.maxLength
        if args.inv is True:
            filter = not filter
        if not filter:
            sys.stdout.write("%s" % str(interval))
        elif args.rename is not None:
            x = copy.deepcopy(interval)
            x.name = args.rename
            sys.stdout.write("%s" % str(x))

    cleanBedTool(tempBedToolPath)
    
if __name__ == "__main__":
    sys.exit(main())
