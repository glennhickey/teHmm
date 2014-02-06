#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import copy

from pybedtools import BedTool, Interval
from teHmm.track import CategoryMap
"""
Add RGB colours to bed file based on state name.
"""


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Assign BED intervals an RGB colour based on their name."
        " A 8-colour palette is used, so if there are more than 8 states the"
        " colours will not be unique...  Make sure itemRgb=On in the track"
        " header before loading in Browser (addTrackHeader.py can do this)")
    parser.add_argument("inBed", help="Input bed file")
    parser.add_argument("outBed", help="Output bed file")
    parser.add_argument("--prefLen", help="Only consider first <PREFLEN>"
                        " bases of each state name when selecting colour.  So"
                        " if the value is <=4, then LTR_Left and LTR_right "
                        "would be assigned the same colour",
                        default=None, type=int)
    parser.add_argument("--col", help="Use given column to determine state"
                        " name.  (Ex. --col=4 will use the name field and"
                        " --col=5 will use score field)", default=4, type=int)
    
    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    outFile = open(args.outBed, "w")
    assert args.col == 4 or args.col == 5

    colourMap = CategoryMap(reserved = 0)

    for interval in BedTool(args.inBed):
        name = interval.name
        if args.col == 5:
            name = interval.score
        if args.prefLen is not None:
            name = name[:args.prefLen]
        colourIdx = colourMap.getMap(name, update = True)
        rgb = palette(colourIdx)
        # cant figure how to add rgb in bed tools, just hack on manually
        toks = str(interval).rstrip().split("\t")
        l = len(toks)
        assert l >= 4
        if l == 4:
            toks.append(".") # score
        if l <= 5:
            toks.append(".") # strand
        if l <= 6:
            toks.append(str(interval.start)) # thickstart
        if l <= 7:
            toks.append(str(interval.end)) # thickend
        if l <= 8:
            toks.append(".") # itemrgb
        assert len(toks) >= 9
        
        toks[8] = ",".join(rgb)
        outFile.write("\t".join(toks))
        outFile.write("\n")

    outFile.close()
        
    
colours = [
#('255', '255', '255'), # white
('0', '0', '0'), #black
('255', '0', '0'), # red
('0', '255', '0'), # green
('0', '0', '255'), # blue
('0', '255', '255'), # cyan
('255', '0', '255'), # magenta
('255', '255', '0'), # yellow
('128', '128', '128'), # gray
]
def palette(idx):
    return colours[idx % len(colours)]

if __name__ == "__main__":
    sys.exit(main())
