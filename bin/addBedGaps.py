#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse

"""
Quick script to make supervised training BED files for the model.  All it does
is add regions that are not covered by known states, together with those that
are, into one giant bed file that covers the entire genome. For example:

ltr.bed : LTR transpoon states that will be used as gold standard
all.bed : an interval for each chormosome (or scaffold) in the genome

addGaps.py all.bed ltr.bed out.bed

will produce a bed file such that all.bed = Union(ltr.bed, out.bed), and each interval in out.bed that is not in ltr.bed will have name 0

"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform add (A-B) to B.  NOTE both input bed files "
        "need to have same number of columns for the script to work in "
        "its present form")
    parser.add_argument("allBed", help="Bed file spanning entire genome")
    parser.add_argument("tgtBed", help="Target intervals")
    parser.add_argument("outBed", help="Output file")
    parser.add_argument("--state", help="state label to add to outpout",
                        default="0")

    args = parser.parse_args()
    assert os.path.isfile(args.allBed)
    assert os.path.isfile(args.tgtBed)
    assert args.outBed != args.allBed and args.outBed != args.tgtBed
    tempFile = "%s_temp" % args.outBed
    tempFile2 = "%s_temp2" % args.outBed

    # make sure that the state label is present in allBed
    os.system("setBedCol.py 3 %s --a %s > %s" % (args.state, args.allBed,
                                                 args.outBed))

    # make sure that there are no overlaps in tgtBed
    os.system("removeBedOverlaps.py %s > %s" % (args.tgtBed, tempFile2))

    # substract tgtbed from state-appended all.bed and store in tempFile
    os.system("subtractBed -a %s -b %s > %s" % (args.outBed, tempFile2,
                                                tempFile))

    # concatenate tempFile to tgtBed and sort into output
    os.system("cat %s %s | sortBed > %s" % (tempFile, tempFile2, args.outBed))

    # remove the temp files.  note that they will trail if anything
    # went wrong above. 
    os.system("rm -f %s" % tempFile)
    os.system("rm -f %s" % tempFile2)
    
    
if __name__ == "__main__":
    sys.exit(main())
