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
        description="Create a teHMM")

    parser.add_argument("inputModel", help="Path of teHMM model created with"
                        " teHmmTrain.py")
    
    args = parser.parse_args()

    hmmModel = TEHMMModel()
    hmmModel.load(args.inputModel)
    print hmmModel.toText()
    
if __name__ == "__main__":
    sys.exit(main())
