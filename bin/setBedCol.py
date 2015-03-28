#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse

from teHmm.common import runShellCommand

"""
Make sure that column X of bed file has value Y
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Set column of bed file, write result to stdout")
    parser.add_argument("col", help="0-based column index to set", type=int)
    parser.add_argument("state", help="state label to add to outpout")
    parser.add_argument("--a", help="path of input file", default="stdin")
    parser.add_argument("--otherBed", help="bedFile (same intervals in same"
                        " order) to obtain values from (overrides state arg)",
                        default=None)
    parser.add_argument("--otherCol", help="0-based column of --otherBed to"
                        " use if specified (by default col arg used)",
                        type=int, default=None)

    args = parser.parse_args()

    ifile = sys.stdin
    if args.a != "stdin":
        ifile = open(args.a)

    otherFile = None
    if args.otherBed is not None:
        otherFile = open(args.otherBed)
        otherCol = args.otherCol
        if otherCol is None:
            otherCol = col
    
    for line in ifile:
        if len(line.lstrip()) == 0 or line.lstrip()[0] == "#":
            sys.stdout.write(line)
        else:
            tokens = line.split()
            ntoks = len(tokens)
            val = args.state
            
            if otherFile is not None:
                otherLine = otherFile.readline()
                while otherLine.lstrip()[0] == "#":
                    otherLine = otherFile.readline()
                otherTokens = otherLine.split()
                if tokens[:3] != otherTokens[:3]:
                    raise RuntimeError("Coordinate mismatch input %s vs "
                                       "other %s" % (str(tokens[:3]),
                                                     str(otherTokens[:3])))
                val = otherTokens[otherCol]
                
            if ntoks > args.col:
                tokens[args.col] = val
            else:
                tokens += [val] * (args.col + 1 - ntoks)
            sys.stdout.write("\t".join(tokens) + "\n")

    if args.a != "stdin":
        ifile.close()
    if otherFile is not None:
        otherFile.close()
    
if __name__ == "__main__":
    sys.exit(main())
