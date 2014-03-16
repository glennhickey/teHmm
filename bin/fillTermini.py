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
Fill pairs of termini.

scaffold_1 0 10 1+
scaffold_2 100 110 1+

becomes

scaffold_1 0 110 1+-1+

Note: DOES NOT WORK ON OUTPUT of cleanTermini
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fill in pairs of termini so there is just one element "
        " reaching from beginning of left termini to end of right termini.")
    parser.add_argument("inBed", help="bed with ltr results to process")
    parser.add_argument("outBed", help="bed to write output to.")
    
    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    outFile = open(args.outBed, "w")

    seen = dict()
    for interval in BedTool(args.inBed).sort():
        if interval.name == "L_Term" or interval.name == "R_Term":
            raise RuntimeError("Need termini IDs.  Cannot run on output of "
                               "cleanTermini.py")
        # Right termini
        if interval.name in seen:
            prevInterval = seen[interval.name]
            del seen[interval.name]
            prevInterval.end = interval.end
            prevInterval.name += "-" + interval.name
            outFile.write(str(prevInterval))
            
        # Left termini
        else:
            seen[interval.name] = interval
            
    outFile.close()
        
if __name__ == "__main__":
    sys.exit(main())
