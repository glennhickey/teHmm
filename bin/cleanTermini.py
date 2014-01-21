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
Replace the ID column with L or R, depending whether its the first or second
instance.

scaffold_1	141	225	1+	43	+
scaffold_1	4479	4563	1+	43	+

becomes

scaffold_1	141	225	L_Term	43	+
scaffold_1	225	4479	R_Term	43	+

"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Change ID to L_term or R_term since the HMM can only use the "
        "two states.")
    parser.add_argument("inBed", help="bed with ltr results to process")
    parser.add_argument("outBed", help="bed to write forward strand output to.")
    parser.add_argument("--rev", help="bed to write reverse strand output to.",
                        default=None)
    
    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    outFile = open(args.outBed, "w")
    outFile2 = None
    if args.rev is not None:
        outFile2 = open(args.rev, "w")

    prevInterval = None
    for interval in BedTool(args.inBed):
        
        origInterval = copy.deepcopy(interval)

        # Right termini
        if prevInterval is not None:
            if interval.name != prevInterval.name:
                raise RuntimeError("Consecutive intervals dont have same id"
                                   "\n%s%s" % (prevInterval, interval))

            interval.name = "R_Term"
            prevInterval = None
            
        # Left termini
        else:
            interval.name = "L_Term"
            prevInterval = origInterval
        
        if origInterval.name[-1] == "+":
            outFile.write(str(interval))
        elif args.rev is not None:
            assert origInterval.name[-1] == "-"
            outFile2.write(str(interval))
                    
    outFile.close()
    if outFile2 is not None:
        outFile2.close()
        
if __name__ == "__main__":
    sys.exit(main())
