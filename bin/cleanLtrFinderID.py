#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse

"""
Take the Bed output of ltr_finder, and make some different bed files that we can use for the model by changing the ids

output.bed : Uniuqe id's removed
output_sym.bed : and right and left removed
output_tsd_as_gap.bed : tsd states removed
output_sym_tsd_as_gap.bed : tsd states removed and right and left removed
output_tsd_as_ltr.bed : tsd states changed to ltr
output_sym_tsd_as_ltr.bed : tsd states changed to ltr right and left removed
output_single.bed : all annotated elems get same id 

"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Remove ltr_finder ids from 4th column")
    parser.add_argument("inBed", help="bed with ltr results to process")
    parser.add_argument("outBed", help="bed to write output to.  Will also "
                        "write outBed_sym.bed outBed_tsd_as_gap.bed etc.")
    
    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    baseOut, ext = os.path.splitext(args.outBed)

    os.system("sed -e \"s/|LTR_TE|[0-9]*//g\" -e \"s/|-//g\" %s > %s" % (
        args.inBed, args.outBed))

    symBed = baseOut + "_sym" + ext
    os.system("sed -e \"s/|left//g\" -e \"s/|right//g\" %s > %s" % (args.outBed,
                                                                    symBed))

    tsd_as_gapsBed = baseOut + "_tsd_as_gap" + ext
    os.system("grep -v TSD %s > %s" % (args.outBed, tsd_as_gapsBed))

    sym_tsd_as_gapsBed = baseOut + "_sym_tsd_as_gap" + ext
    os.system("grep -v TSD %s > %s" % (symBed, sym_tsd_as_gapsBed))

    tsd_as_ltrBed = baseOut + "_tsd_as_ltr" + ext
    os.system("sed -e \"s/TSD/LTR/g\" %s > %s" % (args.outBed, tsd_as_ltrBed))

    sym_tsd_as_ltrBed = baseOut + "_sym_tsd_as_ltr" + ext
    os.system("sed -e \"s/TSD/LTR/g\" %s > %s" % (symBed, sym_tsd_as_ltrBed))

    singleBed = baseOut + "_single" + ext
    os.system("sed -e \"s/LTR/inside/g\" %s > %s" % (sym_tsd_as_ltrBed,
                                                     singleBed))
        
if __name__ == "__main__":
    sys.exit(main())
