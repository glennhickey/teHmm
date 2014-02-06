#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import re

from pybedtools import BedTool, Interval

"""
Remove everything past the first | and / in the name column
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cut names off at first | and or /")
    parser.add_argument("inBed", help="bed with chaux results to process")
    parser.add_argument("outBed", help="bed to write output to.")
    parser.add_argument("--keepSlash", help="dont strip anything after slash "
                        "ex: DNA/HELITRONY1C -> DNA", action="store_true",
                        default=False)
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

    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    outFile = open(args.outBed, "w")

    for interval in BedTool(args.inBed).sort():
        prefix = findPrefix(interval.name, args.mapPrefix)
        if prefix is not None:
            # prefix was specified with --mapPrefix, that's what we use
            interval.name = prefix
        else:
            # otherwise, strip after |
            interval.name = interval.name[:interval.name.find("|")]
            # strip after "/" unless told not to
            if "/" in interval.name and args.keepSlash is False:
                interval.name = interval.name[:interval.name.find("/")]
            # strip trailing digits (and anything after) unless told not to
            if args.leaveNumbers is False:
                m = re.search("\d", interval.name)
                if m is not None:
                    interval.name = interval.name[:m.start()]
        
        outFile.write(str(interval))

    outFile.close()

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
