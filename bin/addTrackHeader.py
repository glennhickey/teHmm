#!/usr/bin/env python
#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import logging
import random
import string

from teHmm.common import runShellCommand
from teHmm.common import runParallelShellCommands
from teHmm.track import TrackList

"""
Add track header to a bed file.  ex:
track	name=hmm7em	description="em 7-state hmm"	useScore=0

We only add when loading a finished bed into the browser because I don't
think the rest of the scripts support an uncommented track line like this.
"""
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Prepend track header onto bed file")

    parser.add_argument("inputBed", help="Path of bed file to add header to")
    parser.add_argument("name", help="Name of track")
    parser.add_argument("description", help="Track description")
    parser.add_argument("--useScore", help="Use score", action="store_true",
                        default=False)
    parser.add_argument("--rgb", help="Enable rgb colours.  These must be "
                        "present in the bed file data (can be added using"
                        "addBedColours.py", action="store_true",
                        default=False)


    args = parser.parse_args()

    # hack together a temporary file path in same directory as input
    S = string.ascii_uppercase + string.digits
    tag = ''.join(random.choice(S) for x in range(5))
    tempPath = os.path.splitext(os.path.basename(args.inputBed))[0] \
                   + "_temp%s.bed" % tag

    score = 0
    if args.useScore is True:
        score = 1

    rgb=""
    if args.rgb is True:
        rgb="\titemRgb=\"On\""
        
    # put the header in the file
    tempFile = open(tempPath, "w")
    tempFile.write("track\tname=\"%s\"\tdescription=\"%s\"\tuseScore=%d%s\n" % (
        args.name, args.description, score, rgb))

    # copy the bed file to the temp file, skipping track header if found
    bedFile = open(args.inputBed, "r")
    skippedTrack = False
    for line in bedFile:
        if skippedTrack == False and len(line) > 11 \
          and line[:11] == "track\tname=":
            skippedTrack = True
        else:
            tempFile.write(line)
    bedFile.close()
    tempFile.close()

    # move the tempfile back to bed file, and hope nothing doesnt go wrong
    runShellCommand("mv %s %s" % (tempPath, args.inputBed))

if __name__ == "__main__":
    sys.exit(main())
