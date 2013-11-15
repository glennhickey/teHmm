#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse

from teHmm.track import TrackData
from teHmm.hmm import MultitrackHmm

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate a given data set with a trained HMM. Display"
        " the log probability of the input data given the model, and "
        "optionally output the most likely sequence of hidden states.")

    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("inputModel", help="Path of hmm created with"
                        "teHmmTrain.py")
    parser.add_argument("chrom", help="Name of sequence (ex chr1, "
                        " scaffold_1) etc.")
    parser.add_argument("start", help="Start position", type=int)
    parser.add_argument("end", help="End position (last plus 1)", type=int)
    parser.add_argument("--bed", help="path of file to write viterbi "
                        "output to (most likely sequence of hidden states)",
                        default=None)
    
    args = parser.parse_args()

    # load model created with teHmmTrain.py
    hmm = MultitrackHmm()
    hmm.load(args.inputModel)

    # load the input
    # read the tracks, while intersecting them with the given interval
    trackData = TrackData()
    # note we pass in the trackList that was saved as part of the model
    # because we do not want to generate a new one.
    trackData.loadTrackData(args.tracksInfo,
                            [(args.chrom, args.start, args.end)],
                            hmm.getTrackList())

    # do the viterbi algorithm
    vitLogProb, vitStates = hmm.viterbi(trackData)[0]

    print "Viterbi (log) score: %f" % vitLogProb

    if args.bed is not None:
        vitOutFile = open(args.bed, "w")
        vitOutFile.write("Viterbi Score: %f\nPath:\n%s\n" % (vitLogProb,
                                                             str(vitStates)))
        
if __name__ == "__main__":
    sys.exit(main())
