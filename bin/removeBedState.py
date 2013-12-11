#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import copy

from pybedtools import BedTool, Interval


"""
Filter out bed intervals that have a given name. Print result to stdout

"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter overlapping intervals out")
    parser.add_argument("inputBed", help="Bed file to filter")
    parser.add_argument("stateName", help="Name of state to filter")
    parser.add_argument("--valCol", help="column to look for state name",
                        type = int, default = 3)
    
    args = parser.parse_args()
    assert os.path.isfile(args.inputBed)
    assert args.valCol == 3 or args.valCol == 4
                        
    bedIntervals = BedTool(args.inputBed).sort()
    prevInterval = None

    for interval in bedIntervals:
        if not ((args.valCol == 3 and interval.name == args.stateName) or
                (args.valCol == 4 and interval.score == args.stateName)):
            sys.stdout.write(str(interval))
    
if __name__ == "__main__":
    sys.exit(main())
