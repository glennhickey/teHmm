#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse

"""
Replace TSD|left|LTR_TE|78 with TSD|left|LTR_TE etc..
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Remove ltr_finder ids from 4th column")
    parser.add_argument("--a", help="path of input file", default="stdin")

    args = parser.parse_args()

    ifile = sys.stdin
    if args.a != "stdin":
        ifile = open(args.a)

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
            if ntoks > 3:
                tokens[3] = tokens[3][:tokens[3].rfind("|")]
            sys.stdout.write("\t".join(tokens) + "\n")

    if args.a != "stdin":
        ifile.close()
        
if __name__ == "__main__":
    sys.exit(main())
