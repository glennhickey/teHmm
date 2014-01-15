#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse

"""
Find candidate target site duplications (TSD's).  These are short *exact* matches
on the forward strand that flank transposable elements (TEs).  This script takes
 as input a BED file identifying candidate TEs, and the genome sequence in FASTA
 format.  Candidate TSDs are searched for immediately before and after each 
 interval in the BED.  Note that contidugous BED intervals will be treated as a
 single interval.  
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Find candidate TSDs (exact forward matches) flanking given"
        "BED intervals")
    parser.add_argument("fastaSequence" help="DNA sequence in FASTA format")
    parser.add_argument("inBed", help="BED file with TEs whose flanking regions "
                        "we wish to search")
    parser.add_argument("outBed", help="BED file containing (only) output TSDs")
    parser.add_argument("--min", help="Minimum length of a TSD",
                        default=4, type=int)
    parser.add_argument("--max", help="Maximum length of a TSD",
                        default=8, type=int)
    parser.add_argument("--all", help="Report all matches in region (as opposed"
                        " to only the nearest to the BED element which is the "
                        "default behaviour", action="store_true", default=False)
    parser.add_argument("--left", help="Number of bases immediately left of the "
                        "BED element to search for the left TSD",
                        default=20, type=int)
    parser.add_argument("--right", help="Number of bases immediately right of "
                        "the BED element to search for the right TSD",
                        default=20, type=int)
    parser.add_argument("--leftName", help="Name of left TSDs in output Bed",
                        default="L_TSD")
    parser.add_argument("--rightName", help="Name of right TSDs in output Bed",
                        default="L_TSD")
    parser.add_argument("--id", help="Assign left/right pairs of TSDs a unique"
                        " matching ID", action="store_true", default=False)
    
    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    
        
if __name__ == "__main__":
    sys.exit(main())
