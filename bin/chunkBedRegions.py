#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import copy

from teHmm.common import intersectSize, initBedTool, cleanBedTool, runShellCommand
from pybedtools import BedTool, Interval

"""
Cut up some bed intervals (probably have already written this somewhere else
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cut up some bed intervals (probably have already "
        "written this somewhere elsee")
    parser.add_argument("inBed", help="Input bed file")
    parser.add_argument("chunk", help="Chunk size", type=int)
    parser.add_argument("--overlap", help="Fraction overlap [0-1]"
                        " between adjacent chunks", type=float, default=0.)
    
    args = parser.parse_args()
    tempBedToolPath = initBedTool()
    assert os.path.exists(args.inBed)
    assert args.overlap >= 0. and args.overlap <= 1.

    wiggle = .2
    step = args.chunk * (1. - args.overlap)

    for interval in BedTool(args.inBed):
        length = interval.end - interval.start
        total = 0
        start = interval.start
        while total < length:
            outInterval = copy.deepcopy(interval)
            assert start < interval.end
            outInterval.start = start
            outInterval.end = start + args.chunk
            if interval.end - outInterval.end < wiggle * args.chunk or\
              outInterval.end > interval.end:
              outInterval.end = interval.end
            total = outInterval.end
            start += step
            sys.stdout.write(str(outInterval))

    cleanBedTool(tempBedToolPath)
            
if __name__ == "__main__":
    sys.exit(main())
