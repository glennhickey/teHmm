#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import copy
import math
from pybedtools import BedTool, Interval
from teHmm.common import EPSILON
"""
Scale numeric values of a given column of a text file.  Quick hack for when
we don't want to have scaling done in tracks.xml file
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scale numeric values of a given column of a text file.")
    parser.add_argument("inFile", help="input text file")
    parser.add_argument("outFile", help="outut text file")
    parser.add_argument("column", help="column (starting at 1) to process",
                        type=int)
    parser.add_argument("--scale", help="compute stats after applying given"
                        " (linear) scale factor (and rounding to int)",
                        type=float, default=None)
    parser.add_argument("--logScale", help="compute stats after applying given"
                        " scale factor to logarithm of each value (and "
                        "rounding to int)", type=float,
                        default=None)
    parser.add_argument("--min", help="only apply scaling to values greater or"
                        " equal to this value", type=float, default=None)
    parser.add_argument("--max", help="only apply scaling to values less or"
                        " equal to this value", type=float, default=None)
    
    
    args = parser.parse_args()
    assert os.path.exists(args.inFile)
    assert args.column > 0
    assert args.scale is None or args.logScale is None
    if args.scale is not None:
        def tform(v):
            return int(float(v) * args.scale)
    elif args.logScale is not None:
        def tform(v):
            return int(math.log(float(v) + EPSILON) * args.logScale)
    else:
        def tform(v):
            return float(v)

    f = open(args.inFile, "r")
    fo = open(args.outFile, "w")
    col = args.column - 1
            
    for line in f:
        try:
            toks = line.split()
            val = float(toks[col])
            if (args.min is None or val >= args.min) and\
              (args.max is None or val <= args.max):
                val = tform(line.split()[col])
            toks[col] = str(val)
            fo.write("\t".join(toks))
            fo.write("\n")
        except:
            fo.write(line)
            continue

    f.close()
    fo.close()
    
        
if __name__ == "__main__":
    sys.exit(main())
