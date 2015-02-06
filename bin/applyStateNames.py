#!/usr/bin/env python

#Copyright (C) 2015 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import numpy as np
from collections import defaultdict

from teHmm.hmm import MultitrackHmm
from teHmm.modelIO import loadModel, saveModel
from teHmm.track import CategoryMap
from teHmm.trackIO import readBedIntervals

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Rename HMM states.")
    parser.add_argument("inputModel", help="Path of teHMM model created with"
                        " teHmmTrain.py")
    parser.add_argument("outputModel", help="Path of model with renamed states")
    parser.add_argument("--newNames", help="comma-separated list of state names to"
                        " apply.  This list must have exactly the same number of"
                        " states as the model.  The ith name in the list will be "
                        "assigned to the ith name of the model...", default=None)
    parser.add_argument("--teNumbers", help="comma-separated list of state numbers"
                        " that will be assigned TE states, with everything else"
                        " assigned Other.  This is less flexible but maybe more"
                        " convenient at times than --newNames.", default=None)
    parser.add_argument("--bed", help="apply naming to bed file and print "
                        "results to stdout", default=None)
    parser.add_argument("--sizes", help="bedFile to use for computing state numbering"
                        " by using decreasing order in total coverage (only works"
                        " with --teNumbers)", default=None)
    parser.add_argument("--noMerge", help="dont merge adjacent intervals with same"
                        " name with --bed option", action="store_true",default=False)
    

    args = parser.parse_args()
    assert args.inputModel != args.outputModel
    assert (args.newNames is None) != (args.teNumbers is None)

    # load model created with teHmmTrain.py
    model = loadModel(args.inputModel)

    # names manually specified
    if args.newNames is not None:
        names = args.newNames.split(",")
        
    # names computed using simple scheme from set of "TE" state numbers (as found from
    # log output of fitStateNames.py)
    elif args.teNumbers is not None:
        teNos = set([int(x) for x in args.teNumbers.split(",")])
        teCount, otherCount = 0, 0
        numStates = model.getEmissionModel().getNumStates()

        # re-order from sizing info
        if args.sizes is not None:
            bedIntervals = readBedIntervals(args.sizes, ncol=4)
            sizeMap = defaultdict(int)
            for interval in bedIntervals:
                sizeMap[int(interval[3])] += interval[2] - interval[1]
            stateNumbers = sorted([x for x in xrange(numStates)],
                           reverse=True, key = lambda x : sizeMap[x])
        else:
            stateNumbers = [x for x in xrange(numStates)]
        names = [""] * numStates
        for i in stateNumbers:
            if i in teNos:
                name = "TE-%d" % teCount
                teCount += 1
            else:
                name = "Other-%d" % otherCount
                otherCount += 1
            names[i] = name
        assert teCount == len(teNos) and teCount + otherCount == len(names)
                
    assert len(names) == model.getEmissionModel().getNumStates()

    # throw names in the mapping object and stick into model
    catMap = CategoryMap(reserved=0)
    for i, name in enumerate(names):
        catMap.getMap(name, update=True)
    model.stateNameMap = catMap
    
    # save model
    saveModel(args.outputModel, model)

    # process optional bed file
    if args.bed is not None:
        prevInterval = None
        bedIntervals = readBedIntervals(args.bed, ncol=4)
        for interval in bedIntervals:
            oldName = interval[3]
            newName = names[int(oldName)]
            newInterval = list(interval)
            newInterval[3] = newName
            if args.noMerge:
                # write interval
                print "\t".join(str(x) for x in newInterval)
            else:
                if prevInterval is None:
                    # update prev interval first time
                    prevInterval = newInterval
                elif newInterval[3] == prevInterval[3] and\
                         newInterval[0] == prevInterval[0] and\
                         newInterval[1] == prevInterval[2]:
                    # glue onto prev interval
                    prevInterval[2] = newInterval[2]
                else:
                    # write and update prev
                    print "\t".join(str(x) for x in prevInterval)
                    prevInterval = newInterval
        if prevInterval is not None:
            print "\t".join(str(x) for x in prevInterval)


if __name__ == "__main__":
    sys.exit(main())
