#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import re

from pybedtools import BedTool, Interval
from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool

"""
Remove everything past the first occurence of | / ? _ in the name column
This is used to clean the names from the various RepeatMasker
(via repeatmasker_to_bed.sh) tracks (formerly called cleanChaux.py)
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cut names off at first |, /, ?, or _")
    parser.add_argument("inBed", help="bed with chaux results to process")
    parser.add_argument("--keepSlash", help="dont strip anything after slash "
                        "ex: DNA/HELITRONY1C -> DNA", action="store_true",
                        default=False)
    parser.add_argument("--keepUnderscore", help="dont strip anything after _ ",
                        action="store_true", default=False)
    parser.add_argument("--leaveNumbers", help="by default, numbers as the end"
                        " of names are trimmed off.  ex: DNA/HELITRONY1C -> "
                        " DNA/HELITRONY. This option disables this behaviour",
                        default=False)
    parser.add_argument("--mapPrefix", help="Rename all strings with given "
                        "prefix to just the prefix. ex: --mapPrefix DNA/HELI"
                        " would cause any instance of DNA/HELITRONY1C or "
                        "HELITRON2 to be mapped to just DNA/HELI.  This option"
                        " overrides --keepSlash and --leaveNumbers for the"
                        " elements to which it applies.  This option can be"
                        " specified more than once. ex --mapPrefix DNA/HELI "
                        "--maxPrefix DNA/ASINE.", action="append")
    parser.add_argument("--minScore", help="Minimum score value to not filter"
                        " out", default=-sys.maxint, type=float)
    parser.add_argument("--maxScore", help="Maximum score value to not filter"
                        " out", default=sys.maxint, type=float)
    parser.add_argument("--overlap", help="Dont run removeBedOverlaps.py",
                        action="store_true", default=False)

    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    assert args.minScore <= args.maxScore
    tempBedToolPath = initBedTool()

    if not args.overlap:
        tempPath1 = getLocalTempPath("Temp1_", ".bed")
        tempPath2 = getLocalTempPath("Temp2_", ".bed")
        runShellCommand("sortBed -i %s > %s" % (args.inBed, tempPath1))
        runShellCommand("removeBedOverlaps.py %s --rm > %s" % (tempPath1,
                                                               tempPath2))
        args.inBed = tempPath2

    for interval in BedTool(args.inBed).sort():
        # filter score if exists
        try:
            if interval.score is not None and\
                (float(interval.score) < args.minScore or
                 float(interval.score) > args.maxScore):
                continue
        except:
            pass
        prefix = findPrefix(interval.name, args.mapPrefix)
        if prefix is not None:
            # prefix was specified with --mapPrefix, that's what we use
            interval.name = prefix
        else:
            # otherwise, strip after |
            if "|" in interval.name:
                interval.name = interval.name[:interval.name.find("|")]
            # strip after ?
            if "?" in interval.name:
                interval.name = interval.name[:interval.name.find("?")]
            #strip after _ unlerss told not to
            if "_" in interval.name and args.keepUnderscore is False:
                interval.name = interval.name[:interval.name.find("_")]
            # strip after "/" unless told not to
            if "/" in interval.name and args.keepSlash is False:
                interval.name = interval.name[:interval.name.find("/")]
            # strip trailing digits (and anything after) unless told not to
            if args.leaveNumbers is False:
                m = re.search("\d", interval.name)
                if m is not None:
                    interval.name = interval.name[:m.start()]
        
        sys.stdout.write(str(interval))
    if not args.overlap:
        runShellCommand("rm -f %s %s" % (tempPath1, tempPath2))
    cleanBedTool(tempBedToolPath)

# should be a prefix tree...
def findPrefix(name, plist):
    if plist is None:
        return None
    for p in plist:
        if name[:len(p)] == p:
            return p
    return None

        
if __name__ == "__main__":
    sys.exit(main())
