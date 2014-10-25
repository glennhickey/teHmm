#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import random
import numpy as np
import time

from teHmm.hmm import MultitrackHmm
from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool
from teHmm.basehmm import BaseHMM, logsumexp
from teHmm.hmm import MultitrackHmm
from teHmm.emission import IndependentMultinomialEmissionModel

""" This is a script to benchmark the HMM dynamic programming functions
(forward, backward and viterbi).  It seems the original scikit implementations
are slower than they need to be.  This script is a sandbox for some tests to
help judge if changes made to the dynamic programming help, without resorting
to using teHmmTrain and teHmmEval which need to load the xml data tracks..
"""
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Benchmark HMM Dynamic Programming Functions.")

    parser.add_argument("--N", help="Number of observations.",
                        type=int, default=200000)
    parser.add_argument("--S", help="Number of states.",
                        type=int, default=10)
    parser.add_argument("--alg", help="Algorithm. Valid options are"
                        " {viterbi, forward, backward}.", default="viterbi")
    parser.add_argument("--new", help="Only run new hmm", action="store_true",
                        default=False)
    parser.add_argument("--old", help="Only run old hmm", action="store_true",
                        default=False)
    parser.add_argument("--fb", help="Run a little forward/backward test",
                        action="store_true", default=False)
    
    args = parser.parse_args()
    alg = args.alg.lower()
    assert alg == "viterbi" or alg == "forward" or alg == "backward"

    # orginal, scikit hmm
    basehmm = BaseHMM(n_components=args.S)
    # new, hopefully faster hmm
    mthmm = MultitrackHmm(emissionModel=IndependentMultinomialEmissionModel(
        args.S, [2]))
    frame = makeFrame(args.S, args.N)
    baseret = None
    mtret = None

    if args.new == args.old or args.old:
        startTime = time.time()
        baseret = runTest(basehmm, frame, alg)
        deltaTime = time.time() - startTime
        print "Elapsed time for OLD %d x %d %s: %s" % (args.N, args.S, args.alg,
                                                       str(deltaTime))
        if args.fb:
            fbTest(basehmm, frame)
             
    if args.new == args.old or args.new:
        startTime = time.time()
        newret = runTest(mthmm, frame, alg)
        deltaTime = time.time() - startTime
        print "Elapsed time for NEW %d x %d %s: %s" % (args.N, args.S, args.alg,
                                                       str(deltaTime))
        if args.fb:
            fbTest(mthmm, frame)
            
    if baseret is not None and mtret is not None:
        # note comparison doesnt mean much since data is so boring so 
        # hopefully hmmTest will be more meaningful.  that said, this will still
        # detect many catastrophic bugs. 
        if alg == "viterbi":
            assert_array_almost_eqal(baseret[0], mtret[0])
            assert_array_eqal(baseret[1], mtret[1])
        else:
            assert_array_almost_equal(baseret, mtret)

    
def makeFrame(numStates, numObs):
    """ Make a dummy framelogprob matrix that the functions execpt.
    probability of a states is just the state # / numSates"""
    frame = np.zeros((numObs, numStates),)
    for i in xrange(len(frame)):
        for j in xrange(numStates):
            frame[i, j] = myLog(float(j) / float(numStates))
            frame[i, j] += myLog((float(i % 9) + 1.) / 10)
    return frame

def runTest(hmm, frame, alg):
    """ call the basehmms low level dp algorithm """
    if alg == "viterbi":
        return hmm._do_viterbi_pass(frame)
    elif alg == "forward":
        return hmm._do_forward_pass(frame)
    elif alg == "backward":
        return hmm._do_backward_pass(frame)
    else:
        assert False

def fbTest(hmm, frame):
    """ check invariants for forward and backward tables """

    #for i in xrange(len(frame)):
    #    frame[i,0] = 0.1
    
    flp, ftable = hmm._do_forward_pass(frame)
    btable = hmm._do_backward_pass(frame)
    bneg = np.zeros((btable.shape[1]))
    for i in xrange(len(bneg)):
        bneg[i] = logsumexp(np.asarray(
            [hmm._log_startprob[j] + frame[0, j] + btable[0, j]\
                             for j in xrange(len(bneg))]))
    blp = logsumexp(bneg)
    print ("FProb = %f  BProb = %f,  delta=%f" % (flp, blp, (flp-blp)))

    # same as above but with some segmentation
    minRatio = 0.01
    maxRatio = 10.
    segRatios = np.zeros((len(frame)))
    random.seed(200)
    for i in xrange(len(segRatios)):
        segRatios[i] = random.uniform(minRatio, maxRatio)

    #segRatios = np.asarray( [0.] * len(segRatios))
    # overload this method to break our random ratios through hmm interface
    def blin(x):
        return segRatios
    if isinstance(hmm, MultitrackHmm):
        hmm.emissionModel.getSegmentRatios = blin
    
    flp, ftable = hmm._do_forward_pass(frame)
    btable = hmm._do_backward_pass(frame)
    bneg = np.zeros((btable.shape[1]))
    for i in xrange(len(bneg)):
        if segRatios[0] > 1.:
            segFac = segRatios[0]-1.
        else:
            segFac = 0.
        bneg[i] = logsumexp(np.asarray(
            [hmm._log_startprob[j] + frame[0, j] + btable[0, j] + \
             hmm._log_transmat[j, j] * segFac \
                             for j in xrange(len(bneg))]))
    blp = logsumexp(bneg)
    print ("SegFProb = %f  SegBProb = %f,  delta=%f" % (flp, blp, (flp-blp)))
    #print np.exp(ftable)
    #print np.exp(btable)
    #print segRatios        
    

if __name__ == "__main__":
    sys.exit(main())

    
