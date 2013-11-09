#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
# Contains some fragments of code from sklearn/tests/test_hmm.py
# (2010 - 2013, scikit-learn developers (BSD License))
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import math
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sklearn.hmm import MultinomialHMM

from teHmm.tracksInfo import TracksInfo
from teHmm.track import *
from teHmm.hmm import MultitrackHmm
from teHmm.emission import IndependentMultinomialEmissionModel

from teHmm.tests.common import getTestDirPath
from teHmm.tests.common import TestBase
from teHmm.tests.bedTrackTest import getTracksInfo
from teHmm.tests.bedTrackTest import getTracksInfoPath


class TestCase(TestBase):

    def setUp(self):
        super(TestCase, self).setUp()
        self.tiPath = self.getTempFilePath()
        ti = getTracksInfo()
        ti.save(self.tiPath)

        ## Copied from MultinomialHMMTestCase in sklearn/tests/test_hmm.py 
        self.prng = np.random.RandomState(9)
        self.n_components = 2   # ('Rainy', 'Sunny')
        self.n_symbols = 3  # ('walk', 'shop', 'clean')
        self.emissionprob = [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]
        self.startprob = [0.6, 0.4]
        self.transmat = [[0.7, 0.3], [0.4, 0.6]]
    
    def tearDown(self):
        super(TestCase, self).tearDown()

    def testInit(self):
        emissionModel = IndependentMultinomialEmissionModel(2, [3])
        hmm = MultitrackHmm(emissionModel)

    def testWikipediaExample(self):
        """ Mostly taken from test_hmm.py in sckikit-learn """
        
        # do scikit model as sanity check
        observations = [0, 1, 2]
        h = MultinomialHMM(self.n_components,
                           startprob=self.startprob,
                           transmat=self.transmat,)
        h.emissionprob_ = self.emissionprob
        logprob, state_sequence = h.decode(observations)
        self.assertAlmostEqual(np.exp(logprob), 0.01344)
        assert_array_equal(state_sequence, [1, 0, 0])

        # do multitrack model (making sure to wrap params in list to reflect
        # extra dimension for tracks)
        trackObs = np.asarray([[0], [1], [2]])
        emissionModel = IndependentMultinomialEmissionModel(2, [3],
                                                            [self.emissionprob])
        trackHmm = MultitrackHmm(emissionModel,
                                 startprob=self.startprob,
                                 transmat=self.transmat)

        # test consistency of log likelihood function
        assert_array_equal(trackHmm._compute_log_likelihood(trackObs),
                           h._compute_log_likelihood(observations))

        # test consistency of viterbi
        logprob, state_sequence = trackHmm.decode(trackObs)
        self.assertAlmostEqual(np.exp(logprob), 0.01344)
        assert_array_equal(state_sequence, [1, 0, 0])

        # add a couple dummy tracks that shouldn't change anything
        trackObs3 = np.asarray([[0,0,0], [1,0,0], [2,0,0]])
        emissionprob3 = [
            [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
            [[1.], [1.]],
            [[1.], [1.]]
            ]
        emissionModel3 = IndependentMultinomialEmissionModel(2, [3,1,1],
                                                             emissionprob3)
        trackHmm3 = MultitrackHmm(emissionModel3,
                                 startprob=self.startprob,
                                 transmat=self.transmat)

         # test consistency of log likelihood function
        assert_array_equal(trackHmm3._compute_log_likelihood(trackObs3),
                           h._compute_log_likelihood(observations))

        # test consistency of viterbi
        logprob, state_sequence = trackHmm3.decode(trackObs3)
        self.assertAlmostEqual(np.exp(logprob), 0.01344)
        assert_array_equal(state_sequence, [1, 0, 0])

         # test consistency of viterbi
        logprob, state_sequence = trackHmm.decode(trackObs)
        self.assertAlmostEqual(np.exp(logprob), 0.01344)
        assert_array_equal(state_sequence, [1, 0, 0])

        # go through same excecise but with another track that has a bunch
        # of equiprobables states
        trackObs4 = np.asarray([[0,0,0,0], [1,0,0,5], [2,0,0,7]])
        emissionprob4 = [
            [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
            [[1.], [1.]],
            [[1.], [1.]],
            [[.1] * 10, [.1] * 10],
            ]
        emissionModel4 = IndependentMultinomialEmissionModel(2, [3,1,1,10],
                                                             emissionprob4)
        trackHmm3 = MultitrackHmm(emissionModel4,
                                 startprob=self.startprob,
                                 transmat=self.transmat)

        # test consistency of viterbi
        logprob, state_sequence = trackHmm3.decode(trackObs4)
        self.assertAlmostEqual(np.exp(logprob), 0.01344 * 0.1 * 0.1 * 0.1)
        assert_array_equal(state_sequence, [1, 0, 0])

        


def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

