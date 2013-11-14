#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse

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

    args = parser.parse_args()

    ifile = sys.stdin
    if args.a != "stdin":
        ifile = open(args.a)
    
    for line in ifile:
        if line.lstrip()[0] == "#":
            sys.stdout.write(line)
        else:
            tokens = line.split()
            ntoks = len(tokens)
            if ntoks > args.col:
                tokens[args.col] = args.state
            else:
                tokens += [args.state] * (args.col + 1 - ntoks)
            sys.stdout.write("\t".join(tokens) + "\n")

    if args.a != "stdin":
        ifile.close()
    
if __name__ == "__main__":
    sys.exit(main())
