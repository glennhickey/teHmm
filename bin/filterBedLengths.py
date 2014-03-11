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
        description="Filter out bed intervals that are smaller than a cutoff")
    parser.add_argument("inputBed", help="Bed file to filter")
    parser.add_argument("minLength", help="Minimum interval length", type=int)
    
    args = parser.parse_args()
    assert os.path.isfile(args.inputBed)
    
    tempBedToolPath = initBedTool()

    bedIntervals = BedTool(args.inputBed).sort()
    
    for interval in bedIntervals:
        if interval.end - interval.start >= args.minLength:
            sys.stdout.write("%s" % str(interval))

    cleanBedTool(tempBedToolPath)
    
if __name__ == "__main__":
    sys.exit(main())
