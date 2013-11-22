#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse

from pybedtools import BedTool, Interval

"""
Replace the ID column with L or R, depending whether its the first or second
instance.
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Change ID to L or R")
    parser.add_argument("inBed", help="bed with ltr results to process")
    parser.add_argument("outBed", help="bed to write output to.")
    
    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    outFile = open(args.outBed, "w")

    prevChrom = None
    seen = set()
    for interval in BedTool(args.inBed).sort():
        if interval.chrom != prevChrom:
            seen = set()
        isMatch = interval.name in seen
        seen.add(interval.name)
        prevChrom = interval.chrom
        if isMatch:
            interval.name = "R_Term"
        else:
            interval.name = "L_Term"
        outFile.write(str(interval))

    outFile.close()
        
if __name__ == "__main__":
    sys.exit(main())
