#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse

from pybedtools import BedTool, Interval

"""
Remove everything past the first | and / in the name column
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cut names off at first | and or /")
    parser.add_argument("inBed", help="bed with chaux results to process")
    parser.add_argument("outBed", help="bed to write output to.")

    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    outFile = open(args.outBed, "w")

    for interval in BedTool(args.inBed).sort():
        interval.name = interval.name[:interval.name.find("|")]
        if "/" in interval.name:
            interval.name = interval.name[:interval.name.find("/")]
        outFile.write(str(interval))

    outFile.close()
        
if __name__ == "__main__":
    sys.exit(main())
