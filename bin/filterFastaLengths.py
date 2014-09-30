#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import copy

from teHmm.trackIO import fastaRead
"""
Filter out fasta sequences that are smaller than a cutoff
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter out fasta sequences that are smaller than a cutoff")
    parser.add_argument("inputFasta", help="Fasta file to filter")
    parser.add_argument("minLength", help="Minimum sequence length", type=int)
    parser.add_argument("--keyword", help="Only filter out sequence if keyword"
                        " is present in its name (case insensitive)", default=None)

    
    args = parser.parse_args()
    assert os.path.isfile(args.inputFasta)
    if args.keyword is not None:
        args.keyword = args.keyword.lower()

    faFile = open(args.inputFasta, "r")
    for seqName, sequence in fastaRead(faFile):
        if len(sequence) >= args.minLength or\
           (args.keyword is not None and \
            args.keyword not in seqName.lower()):
            print ">" + seqName
            numLines = len(sequence) / 50
            if len(sequence) % 50 > 0:
                numLines += 1
            printed = 0
            for i in xrange(numLines):
                print sequence[i * 50: i * 50 + 50]
                printed += len(sequence[i * 50: i * 50 + 50])
            assert printed == len(sequence)
    faFile.close()
    
if __name__ == "__main__":
    sys.exit(main())
