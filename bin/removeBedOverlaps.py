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
Filter out bed intervals that overlap other intervals.

Algorithm:  for each interval in sorted list, cut (or remove) such that it
doesn't overlap any intervals before it in the list. 
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter overlapping intervals out")
    parser.add_argument("inputBed", help="Bed file to filter")
    parser.add_argument("--bed12", help="Use bed12 exons instead of start/end"
                        " if present (equivalent to running bed12ToBed6 on"
                        " input first).", action="store_true", default=False)
    
    args = parser.parse_args()
    assert os.path.isfile(args.inputBed)

    bedIntervals = BedTool(args.inputBed).sort()
    if args.bed12 is True:
        bedIntervals = bedIntervals.bed6()
        
    prevInterval = None

    # this code has been way to buggy for something so simple
    # keep extra list to check for sure even though it's a waste of
    # time and space
    sanity = []
    
    for interval in bedIntervals:
        if (prevInterval is not None and
            interval.chrom == prevInterval.chrom and
            interval.start < prevInterval.end):
            interval.start = prevInterval.end
            
        if interval.end > interval.start:
            sys.stdout.write("%s" % str(interval))
            sanity.append(interval)
            prevInterval = interval

    for i in xrange(len(sanity) - 1):
        if sanity[i].chrom == sanity[i+1].chrom:
            assert sanity[i+1].start >= sanity[i].end
    
if __name__ == "__main__":
    sys.exit(main())
