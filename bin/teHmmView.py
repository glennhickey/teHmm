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
        description="Create a teHMM")

    parser.add_argument("inputModel", help="Path of teHMM model created with"
                        " teHmmTrain.py")
    
    args = parser.parse_args()

    # load model created with teHmmTrain.py
    hmm = MultitrackHmm()
    hmm.load(args.inputModel)

    # crappy print method
    print hmm.toText()
    
if __name__ == "__main__":
    sys.exit(main())
