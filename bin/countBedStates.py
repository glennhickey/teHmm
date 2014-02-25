#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
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
Count the number of unique bed states in file.
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute number of unique BED names in file")
    parser.add_argument("inFile", help="input text file")
    parser.add_argument("--col", help="column (starting at 1) to process",
                        type=int, default=4)
    
    args = parser.parse_args()
    assert os.path.exists(args.inFile)

    f = open(args.inFile, "r")
    col = args.col - 1
    numLines = 0
    s = set()
    
    for line in f:
        c = line.lstrip()
        if len(c) > 0 and c[0] != "#":
            toks = c.split()
            if len(toks) > col:
                val = toks[col]
                s.add(val)
                numLines += 1
                    
    f.close()

    print "Number of states (unique): %d %d" % (numLines, len(s))
        
if __name__ == "__main__":
    sys.exit(main())
