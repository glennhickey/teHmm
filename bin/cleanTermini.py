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

Optionally split into up to for files depending on strand and side
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter bed of lastz termini for use with HMM.  By default "
        "unique id in name column is changed to L_Term or R_Term")
    parser.add_argument("inBed", help="bed with ltr results to process")
    parser.add_argument("outBed", help="bed to write forward strand output to.")
    parser.add_argument("--splitStrand", help="write forwards strand to <outBed>_f and"
                        " reverse strand to <outBed>_b", action="store_true",
                        default=False)
    parser.add_argument("--splitSide", help="write left termini to <outBed>_l and"
                        " right side to <outBed>_r", action="store_true",
                        default=False)
    parser.add_argument("--leaveName", help="dont change the name column",
                        action="store_true", default=False)
    
    args = parser.parse_args()
    assert os.path.exists(args.inBed)

    # create the 4 file handles for the output. depending on
    # args some of them can actually be same. 
    name, ext = os.path.splitext(args.outBed)
    f, b, l, r = "","","",""
    if args.splitSide is True:
        l, r = "_l", "_r"
    if args.splitStrand is True:
        f, b = "_f", "_b"
    lfPath = "%s%s%s%s" % (name, l, f, ext)
    rfPath = "%s%s%s%s" % (name, r, f, ext)
    lbPath = "%s%s%s%s" % (name, l, b, ext)
    rbPath = "%s%s%s%s" % (name, r, b, ext)
    files = dict()
    def getFile(path):
        if path not in files:
            assert path != args.inBed
            file = open(path, "w")
            files[path] = file            
        return files[path]
    lfFile = getFile(lfPath)
    rfFile = getFile(rfPath)
    lbFile = getFile(lbPath)
    rbFile = getFile(rbPath)

    prevInterval = None
    for interval in BedTool(args.inBed):
        
        origInterval = copy.deepcopy(interval)
        # Right termini
        if prevInterval is not None:
            if interval.name != prevInterval.name:
                raise RuntimeError("Consecutive intervals dont have same id"
                                   "\n%s%s" % (prevInterval, interval))

            if args.leaveName is False:
                interval.name = "R_Term"
            prevInterval = None
            left = False
            
        # Left termini
        else:
            if args.leaveName is False:
                interval.name = "L_Term"
            prevInterval = origInterval
            left = True

        forward = origInterval.name[-1] == "+"

        file = None
        if left:
            if forward:
                file = lfFile
            else:
                file = lbFile
        else:
            if forward:
                file = rfFile
            else:
                file = rbFile
        file.write(str(interval))

    cset = set()
    for file in [lfFile, rfFile, lbFile, rbFile]:
        if file not in cset:
            file.close()
            cset.add(file)
        
if __name__ == "__main__":
    sys.exit(main())
