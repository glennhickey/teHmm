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
Extract runs of a single nucleotide out of a FASTA file and into a BED file
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract runs of a single nucleotide from a "
        "FASTA file into a BED file.")
    parser.add_argument("inputFa", help="Input FASTA file")
    parser.add_argument("minLength", help="Minimum interval length", type=int)
    parser.add_argument("--caseSensitive", help="Case sensitive comparison",
                        action="store_true", default=False)
    parser.add_argument("--nuc", help="Nucleotide to consider",
                        default="N")
    
    args = parser.parse_args()
    assert os.path.isfile(args.inputFa)
    assert len(args.nuc) == 1
    faFile = open(args.inputFa, "r")
    if args.caseSensitive is False:
        args.nuc = args.nuc.upper()
        
    for seqName, seqString in fastaRead(faFile):
        curInterval = [None, -2, -2]
        for i in xrange(len(seqString)):
            
            if seqString[i] == args.nuc or (args.caseSensitive is False and
                                            seqString[i].upper() == args.nuc):
                # print then re-init curInterval
                if i != curInterval[2]:
                    if curInterval[0] is not None:
                        printInterval(sys.stdout, curInterval, args)
                    curInterval = [seqName, i, i + 1]
                # extend curInterval
                else:
                    assert seqName == curInterval[0]
                    assert i > curInterval[1]
                    curInterval[2] = i + 1

            # print last interval if exists
            if i == len(seqString) - 1 and curInterval[0] is not None:
                printInterval(sys.stdout, curInterval, args)
                
    faFile.close()


def printInterval(ofile, curInterval, args):
    l = curInterval[2] - curInterval[1]
    if l >= args.minLength:
        ofile.write("%s\t%d\t%d\t%s\t%d\n" % (
            curInterval[0],
            curInterval[1],
            curInterval[2],
            args.nuc,
            l))
    
if __name__ == "__main__":
    sys.exit(main())
