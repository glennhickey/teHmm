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
Get some basic statistics about *NUMERIC* values in a particular column of a
text file (ex BED or WIG).  Non-numeric values are ignored. Information is
used for setting binning parameters in the track xml file
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute statistics for *NUMERIC* value column of text"
        " file (ex bed or wig)")
    parser.add_argument("inFile", help="input text file")
    parser.add_argument("column", help="column (starting at 1) to process",
                        type=int)
    
    args = parser.parse_args()
    assert os.path.exists(args.inFile)
    assert args.column > 0

    f = open(args.inFile, "r")
    col = args.column - 1
    minVal = sys.maxint
    maxVal = -sys.maxint
    valDict = dict()
    numVals = 0
    numLines = 0
    total = 0.0
    for line in f:
        numLines += 1
        try:
            val = float(line.split()[col])
            minVal = min(val, minVal)
            maxVal = max(val, maxVal)
            numVals += 1
            valDict[val] = 1
            total += val
        except:
            continue

    if numVals == 0 and numLines > 0:
        raise RuntimeError("Zero values read out of %d lines.  Perhaps column "
                           "%d doesnt exist or is not numeric" % (
                               numLines, args.column))
    numVals = float(numVals)
    meanVal = total / numVals
    numUnique = len(valDict)

    f.seek(0)
    dTotal = 0.0
    for line in f:
        try:
            val = float(line.split()[col])
            dTotal += (val - meanVal) * (val - meanVal)
        except:
            continue
    variance = dTotal / numVals

    print "Number of values (unique): %d (%d)" % (numVals, numUnique)
    print "Min Max: %f %f" % (minVal, maxVal)
    print "Mean Variance: %f %f" % (meanVal, variance)

    f.close()
    
        
if __name__ == "__main__":
    sys.exit(main())
