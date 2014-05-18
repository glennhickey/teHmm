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

from teHmm.basehmm import MultinomialHMM

from teHmm.track import *
from teHmm.trackIO import readBedIntervals
from teHmm.hmm import MultitrackHmm
from teHmm.emission import IndependentMultinomialEmissionModel

from teHmm.tests.common import getTestDirPath
from teHmm.tests.common import TestBase
from teHmm.tests.bedTrackTest import getTracksInfoPath
from teHmm.tests.emissionTest import getBedStates

class TestCase(TestBase):

    def setUp(self):
        super(TestCase, self).setUp()

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
        emissionModel = IndependentMultinomialEmissionModel(
            2, [3], zeroAsMissingData=False)
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
        emissionModel = IndependentMultinomialEmissionModel(
            2, [3], [self.emissionprob], zeroAsMissingData=False)
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
        emissionModel3 = IndependentMultinomialEmissionModel(
            2, [3,1,1], emissionprob3, zeroAsMissingData=False)
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
        emissionModel4 = IndependentMultinomialEmissionModel(
            2, [3,1,1,10], emissionprob4, zeroAsMissingData=False)
        trackHmm3 = MultitrackHmm(emissionModel4,
                                 startprob=self.startprob,
                                 transmat=self.transmat)

        # test consistency of viterbi
        logprob, state_sequence = trackHmm3.decode(trackObs4)
        self.assertAlmostEqual(np.exp(logprob), 0.01344 * 0.1 * 0.1 * 0.1)
        assert_array_equal(state_sequence, [1, 0, 0])

        # make sure that it still works using a TrackTable structure instead
        # of a numpy array
        trackTable4 = IntegerTrackTable(4, "scaffold_1", 10, 13)
        for row in xrange(4):
            trackTable4.writeRow(row, [trackObs4[0][row],
                                       trackObs4[1][row],
                                       trackObs4[2][row]])
        logprob, state_sequence = trackHmm3.decode(trackTable4)
        self.assertAlmostEqual(np.exp(logprob), 0.01344 * 0.1 * 0.1 * 0.1)
        assert_array_equal(state_sequence, [1, 0, 0])
        

    def testPredict(self):
        observations = [0, 1, 2]
        h = MultinomialHMM(self.n_components,
                           startprob=self.startprob,
                           transmat=self.transmat,)
        h.emissionprob_ = self.emissionprob
        state_sequence = h.predict(observations)
        posteriors = h.predict_proba(observations)
        assert_array_equal(state_sequence, [1, 0, 0])
        assert_array_almost_equal(posteriors, [
            [0.23170303, 0.76829697],
            [0.62406281, 0.37593719],
            [0.86397706, 0.13602294],
        ])

         # add a couple dummy tracks that shouldn't change anything
        trackObs3 = np.asarray([[0,0,0], [1,0,0], [2,0,0]])
        emissionprob3 = [
            [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
            [[1.], [1.]],
            [[1.], [1.]]
            ]
        emissionModel3 = IndependentMultinomialEmissionModel(
            2, [3,1,1], emissionprob3, zeroAsMissingData=False)
        trackHmm3 = MultitrackHmm(emissionModel3,
                                  startprob=self.startprob,
                                  transmat=self.transmat)
        state_sequence = trackHmm3.predict(trackObs3)
        posteriors = trackHmm3.predict_proba(trackObs3)
        assert_array_equal(state_sequence, [1, 0, 0])
        
        #assert_array_almost_equal(posteriors, [
        #    [0.23170303, 0.76829697],
        #    [0.62406281, 0.37593719],
        #    [0.86397706, 0.13602294],
        #])
        # above is no longer true since we fixed bacwkardTable[N] to be a dsitrubtion
        # rather than just all 1s.  instead we do a test to make sure that total
        # probability from forward is the same as from backward
        emProbs = emissionModel3.allLogProbs(trackObs3)
        flp, ftable = trackHmm3._do_forward_pass(emProbs)
        emProbsOld = h._compute_log_likelihood(np.asarray(observations))
        flpOld, ftableOld = h._do_forward_pass(emProbsOld)
        assert_array_almost_equal(ftable, ftableOld)
        assert_array_almost_equal(flp, flpOld)
        
        btable = trackHmm3._do_backward_pass(emProbs)
        bneg1 = np.zeros((self.n_components))
        for i in xrange(self.n_components):
            for j in xrange(self.n_components):
                bneg1[i] += np.exp(trackHmm3._log_startprob[j] + emProbs[0, j] +\
                                  btable[0, j])

        assert np.log(np.sum(bneg1)) == flp
            
    

    def testFit(self):
        h = MultinomialHMM(self.n_components,
                           startprob=self.startprob,
                           transmat=self.transmat)
                           
        h.emissionprob_ = self.emissionprob
        train_obs = [h.sample(n=10)[0] for x in range(10)]
        train_obs3 = []
        for o in train_obs:
            o3 = np.empty((len(o), 3), dtype=np.float)
            for i in xrange(len(o)):
                o3[i][0] = o[i]
                o3[i][1] = 0
                o3[i][2] = 0
            train_obs3.append(o3)
        
        for params in ["s", "t", "e", "st", "se", "te", "ste"]:
            # dont randomly initialize emission model for now since
            # our class doesnt support it yet
            init_params = params.replace("e", "")
            hTrain = MultinomialHMM(self.n_components, params=params,
                                    init_params=init_params)
            hTrain.transmat_ = [[0.5, 0.5], [0.5, 0.5]]
            hTrain._set_emissionprob([[1./3., 1./3., 1./3.],
                                      [1./3., 1./3., 1./3.]])
            hTrain.startprob_ = [0.5, 0.5]
            hTrain.fit(train_obs)
            
            emissionModel3 = IndependentMultinomialEmissionModel(
                2, [3,1,1], zeroAsMissingData=False)
            trackHmm3 = MultitrackHmm(emissionModel3, params=params,
                                      init_params=init_params)
            trackHmm3.transmat_ = [[0.5, 0.5], [0.5, 0.5]]
            trackHmm3.startprob_ = [0.5, 0.5]

            trackHmm3.fit(train_obs3)
            
            assert (hTrain.n_iter == trackHmm3.n_iter)
            assert (hTrain.thresh == trackHmm3.thresh)
            assert_array_equal(hTrain.transmat_, trackHmm3.transmat_)
            for state in xrange(2):
                for symbol in xrange(3):
                    ep = hTrain.emissionprob_[state][symbol]
                    ep3 = trackHmm3.emissionModel.singleLogProb(
                                state, np.asarray([symbol, 0, 0]))
                    assert_array_almost_equal(ep, np.exp(ep3))

            # test consistency of log likelihood function
            assert_array_equal(trackHmm3._compute_log_likelihood(train_obs3[0]),
                               hTrain._compute_log_likelihood(train_obs[0]))

            # test consistency of viterbi
            logprob, state_sequence = hTrain.decode(train_obs[0])
            logprob3, state_sequence3 = trackHmm3.decode(train_obs3[0])
            self.assertAlmostEqual(logprob, logprob3)
            assert_array_equal(state_sequence, state_sequence3)

        
    def testSupervisedLearn(self):
        intervals = readBedIntervals(getTestDirPath("truth.bed"), ncol=4)
        truthIntervals = []
        for i in intervals:
            truthIntervals.append((i[0], i[1], i[2], int(i[3])))

        allIntervals = [(truthIntervals[0][0],
                        truthIntervals[0][1],
                        truthIntervals[-1][2])]
        trackData = TrackData()
        trackData.loadTrackData(getTracksInfoPath(3), allIntervals)
        assert len(trackData.getTrackTableList()) == 1
        # set the fudge to 1 since when the test was written this was
        # hardcoded default
        em = IndependentMultinomialEmissionModel(
            4, trackData.getNumSymbolsPerTrack(),
			  fudge = 1.0)
        hmm = MultitrackHmm(em)
        hmm.supervisedTrain(trackData, truthIntervals)
        hmm.validate()

        # check emissions, they should basically be binary. 
        trackList = hmm.getTrackList()
        emp = np.exp(em.getLogProbs())
        ltrTrack = trackList.getTrackByName("ltr")
        track = ltrTrack.getNumber()
        cmap = ltrTrack.getValueMap()
        s0 = cmap.getMap(None)
        s1 = cmap.getMap(0)
        # we add 1 to all frequencies like emission trainer
        assert_array_almost_equal(emp[track][0][s0], 36. / 37.) 
        assert_array_almost_equal(emp[track][0][s1], 1 - 36. / 37.)
        assert_array_almost_equal(emp[track][1][s0], 1 - 6. / 7.) 
        assert_array_almost_equal(emp[track][1][s1], 6. / 7.)
        assert_array_almost_equal(emp[track][2][s0], 26. / 27.) 
        assert_array_almost_equal(emp[track][2][s1], 1. - 26. / 27.)
        assert_array_almost_equal(emp[track][3][s0], 1. - 6. / 7.)
        assert_array_almost_equal(emp[track][3][s1], 6. / 7.)

        insideTrack = trackList.getTrackByName("inside")
        track = insideTrack.getNumber()
        cmap = insideTrack.getValueMap()
        s0 = cmap.getMap(None)
        s1 = cmap.getMap("Inside")
        assert_array_almost_equal(emp[track][0][s0], 36. / 37.) 
        assert_array_almost_equal(emp[track][0][s1], 1 - 36. / 37.)
        assert_array_almost_equal(emp[track][1][s0], 6. / 7.)
        assert_array_almost_equal(emp[track][1][s1], 1 - 6. / 7.)
        assert_array_almost_equal(emp[track][2][s0], 1. - 26. / 27.)
        assert_array_almost_equal(emp[track][2][s1], 26. / 27.) 
        assert_array_almost_equal(emp[track][3][s0], 6. / 7.)
        assert_array_almost_equal(emp[track][3][s1], 1. - 6. / 7.)

        
        # crappy check for start probs.  need to test transition too!
        freq = [0.0] * em.getNumStates()
        total = 0.0
        for interval in truthIntervals:
           state = interval[3]
           freq[state] += float(interval[2]) - float(interval[1])
           total += float(interval[2]) - float(interval[1])

        sprobs = hmm.getStartProbs()
        assert len(sprobs) == em.getNumStates()
        for state in xrange(em.getNumStates()):
            assert_array_almost_equal(freq[state] / total, sprobs[state])

        # transition probabilites
        # from eyeball:
        #c	0	5	0   0->0 +4   0->1 +1    0-> +5
        #c	5	10	1   1->1 +4   1->2 +1    1-> +5
        #c	10	35	2   2->2 +24  2->3 +1    2-> +25
        #c	35	40	3   3->3 +4   3->0 +1    3-> +5
        #c	40	70	0   0->0 +29             0-> +19
        realTransProbs = np.array([
            [33. / 34., 1. / 34., 0., 0.],
            [0., 4. / 5., 1. / 5., 0.],
            [0., 0., 24. / 25., 1. / 25.],
            [1. / 5., 0., 0., 4. / 5.]
            ])
            
        tprobs = hmm.getTransitionProbs()
        assert tprobs.shape == (em.getNumStates(), em.getNumStates())
        assert_array_almost_equal(tprobs, realTransProbs)
        prob, states = hmm.viterbi(trackData)[0]
        for truthInt in truthIntervals:
            for i in xrange(truthInt[1], truthInt[2]):
                assert states[i] == truthInt[3]

def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

