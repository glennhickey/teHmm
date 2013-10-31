#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse

from teHmm.tracksInfo import TracksInfo
from teHmm.teHmmModel import TEHMMModel


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
    parser.add_argument("sequence", help="Name of sequence (ex chr1, "
                        " scaffold_1) etc.")
    parser.add_argument("start", help="Start position", type=int)
    parser.add_argument("end", help="End position (last plus 1)", type=int)
    parser.add_argument("--viterbi", help="path of file to write viterbi "
                        "output to (most likely sequence of hidden states)",
                        default=None)
    
    args = parser.parse_args()

    hmmModel = TEHMMModel()
    hmmModel.load(args.inputModel)
    hmmModel.loadTrackData(args.tracksInfo, args.sequence, args.start,
                           args.end, False)
    print hmmModel.score()

    if args.viterbi is not None:
        prob, states = hmmModel.viterbi()
        vitOutFile = open(args.viterbi, "w")
        vitOutFile.write("Viterbi Score: %f\nPath:\n%s\n" % (prob,str(states)))
        
if __name__ == "__main__":
    sys.exit(main())
