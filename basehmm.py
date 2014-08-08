#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt

"""
Moving the multinomial hmm from Scikit-Learn into this package so I can speed up the dynamic programming.  Also take the opportunity to cut off the dependenciy to scikit which was a bit of an overkill since the basic hmm we use is only a fe w lines of code.  Bordering on a total rewrite here, which is probably for the best since the hmm seems on it's way out of scikit anyway.  

-- Glenn Hickey, 2014

 Derived from scikit-learn/sklearn/hmm.py
 See below:

Copyright (c) 2007-2014 the scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

"""# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
# and Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>

"""
The :mod:`sklearn.hmm` module implements hidden Markov models.

**Warning:** :mod:`sklearn.hmm` is orphaned, undocumented and has known
numerical stability issues. This module will be removed in version 0.17.
"""

import string
import numpy as np
import numbers
import copy

from . import _basehmm
from .common import logger

ZEROLOGPROB = -1e200
EPS = np.finfo(float).eps
NEGINF = -np.inf
decoder_algorithms = ("viterbi", "map")

def logsumexp(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.utils.extmath import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def normalize(A, axis=None):
    """ Normalize the input array so that it sums to 1.

    WARNING: The HMM module and its functions will be removed in 0.17
    as it no longer falls within the project's scope and API.

    Parameters
    ----------
    A: array, shape (n_samples, n_features)
       Non-normalized input data
    axis: int
          dimension along which normalization is performed

    Returns
    -------
    normalized_A: array, shape (n_samples, n_features)
        A with values normalized (summing to 1) along the prescribed axis

    WARNING: Modifies inplace the array
    """
    A += EPS
    Asum = A.sum(axis)
    if axis and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape
    return A / Asum

class BaseHMM(object):
    """Hidden Markov Model base class.

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    See the instance documentation for details specific to a
    particular object.

    .. warning::

       The HMM module and its functions will be removed in 0.17
       as it no longer falls within the project's scope and API.

    Attributes
    ----------
    n_components : int
        Number of states in the model.

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    transmat_prior : array, shape (`n_components`, `n_components`)
        Matrix of prior transition probabilities between states.

    startprob_prior : array, shape ('n_components`,)
        Initial state occupation prior distribution.

    algorithm : string, one of the decoder_algorithms
        decoder algorithm

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, and other characters for subclass-specific
        emmission parameters. Defaults to all parameters.


    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emmission parameters. Defaults to all
        parameters.

    See Also
    --------
    GMM : Gaussian mixture model
    """

    # This class implements the public interface to all HMMs that
    # derive from it, including all of the machinery for the
    # forward-backward and Viterbi algorithms.  Subclasses need only
    # implement _generate_sample_from_state(), _compute_log_likelihood(),
    # _init(), _initialize_sufficient_statistics(),
    # _accumulate_sufficient_statistics(), and _do_mstep(), all of
    # which depend on the specific emission distribution.
    #
    # Subclasses will probably also want to implement properties for
    # the emission distribution parameters to expose them publicly.

    def __init__(self, n_components=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, thresh=1e-2, params=string.ascii_letters,
                 init_params=string.ascii_letters):

        self.n_components = n_components
        self.n_iter = n_iter
        self.thresh = thresh
        self.params = params
        self.init_params = init_params
        self.startprob_ = startprob
        self.startprob_prior = startprob_prior
        self.transmat_ = transmat
        self.transmat_prior = transmat_prior
        self._algorithm = algorithm
        self.random_state = random_state

    def eval(self, X):
        return self.score_samples(X)

    def score_samples(self, obs):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        logprob : float
            Log likelihood of the sequence ``obs``.

        posteriors : array_like, shape (n, n_components)
            Posterior probabilities of each state for each
            observation

        See Also
        --------
        score : Compute the log probability under the model
        decode : Find most likely state sequence corresponding to a `obs`
        """
        obs = np.asarray(obs)
        framelogprob = self._compute_log_likelihood(obs)
        logprob, fwdlattice = self._do_forward_pass(framelogprob)
        bwdlattice = self._do_backward_pass(framelogprob)
        gamma = fwdlattice + bwdlattice
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
        posteriors += np.finfo(np.float32).eps
        posteriors /= np.sum(posteriors, axis=1).reshape((-1, 1))
        return logprob, posteriors

    def score(self, obs):
        """Compute the log probability under the model.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : float
            Log likelihood of the ``obs``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors

        decode : Find most likely state sequence corresponding to a `obs`
        """
        obs = np.asarray(obs)
        framelogprob = self._compute_log_likelihood(obs)
        logprob, _ = self._do_forward_pass(framelogprob)
        return logprob

    def _decode_viterbi(self, obs):
        """Find most likely state sequence corresponding to ``obs``.

        Uses the Viterbi algorithm.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        viterbi_logprob : float
            Log probability of the maximum likelihood path through the HMM.

        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model
        """
        framelogprob = self._compute_log_likelihood(np.asarray(obs))
        viterbi_logprob, state_sequence = self._do_viterbi_pass(framelogprob,
                                                                obs = obs)
        return viterbi_logprob, state_sequence

    def _decode_map(self, obs):
        """Find most likely state sequence corresponding to `obs`.

        Uses the maximum a posteriori estimation.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        map_logprob : float
            Log probability of the maximum likelihood path through the HMM
        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        score : Compute the log probability under the model.
        """
        _, posteriors = self.score_samples(obs)
        state_sequence = np.argmax(posteriors, axis=1)
        map_logprob = np.max(posteriors, axis=1).sum()
        return map_logprob, state_sequence

    def decode(self, obs, algorithm="viterbi"):
        """Find most likely state sequence corresponding to ``obs``.
        Uses the selected algorithm for decoding.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        algorithm : string, one of the `decoder_algorithms`
            decoder algorithm to be used

        Returns
        -------
        logprob : float
            Log probability of the maximum likelihood path through the HMM

        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model.
        """
        if self._algorithm in decoder_algorithms:
            algorithm = self._algorithm
        elif algorithm in decoder_algorithms:
            algorithm = algorithm
        decoder = {"viterbi": self._decode_viterbi,
                   "map": self._decode_map}
        logprob, state_sequence = decoder[algorithm](obs)
        return logprob, state_sequence

    def predict(self, obs, algorithm="viterbi"):
        """Find most likely state sequence corresponding to `obs`.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation
        """
        _, state_sequence = self.decode(obs, algorithm)
        return state_sequence

    def predict_proba(self, obs):
        """Compute the posterior probability for each state in the model

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        T : array-like, shape (n, n_components)
            Returns the probability of the sample for each state in the model.
        """
        _, posteriors = self.score_samples(obs)
        return posteriors

    def sample(self, n=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        random_state: RandomState or an int seed (0 by default)
            A random number generator instance. If None is given, the
            object's random_state is used

        Returns
        -------
        (obs, hidden_states)
        obs : array_like, length `n` List of samples
        hidden_states : array_like, length `n` List of hidden states
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        startprob_pdf = self.startprob_
        startprob_cdf = np.cumsum(startprob_pdf)
        transmat_pdf = self.transmat_
        transmat_cdf = np.cumsum(transmat_pdf, 1)

        # Initial state.
        rand = random_state.rand()
        currstate = (startprob_cdf > rand).argmax()
        hidden_states = [currstate]
        obs = [self._generate_sample_from_state(
            currstate, random_state=random_state)]

        for _ in range(n - 1):
            rand = random_state.rand()
            currstate = (transmat_cdf[currstate] > rand).argmax()
            hidden_states.append(currstate)
            obs.append(self._generate_sample_from_state(
                currstate, random_state=random_state))

        return np.array(obs), np.array(hidden_states, dtype=int)

    def fit(self, obs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation.

        Notes
        -----
        In general, `logprob` should be non-decreasing unless
        aggressive pruning is used.  Decreasing `logprob` is generally
        a sign of overfitting (e.g. a covariance parameter getting too
        small).  You can fix this by getting more training data,
        or strengthening the appropriate subclass-specific regularization
        parameter.
        """

        if self.algorithm not in decoder_algorithms:
            self._algorithm = "viterbi"

        self._init(obs, self.init_params)

        logprob = []
        for i in range(copy.deepcopy(self.n_iter)):
            # Expectation step
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for seq in obs:
                framelogprob = self._compute_log_likelihood(seq)
                lpr, fwdlattice = self._do_forward_pass(framelogprob, obs = seq)
                bwdlattice = self._do_backward_pass(framelogprob, obs = seq)
                logger.debug("Computing posteriors from forward/backward"
                             " tables. (last unoptimized part that needs"
                             " redoing for both time and memory)")
                gamma = fwdlattice + bwdlattice
                posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
                logger.debug("Done posteriors")
                curr_logprob += lpr
                self._accumulate_sufficient_statistics(
                    stats, seq, framelogprob, posteriors, fwdlattice,
                    bwdlattice, self.params)
            logprob.append(curr_logprob)
            logMsgString = "BW Iteration %d: LogProb %f" % (i, curr_logprob)
            if i > 0:
                logMsgString += " (delta %f)" % (logprob[-1] - logprob[-2])
            logger.info(logMsgString)

            # Check for convergence.
            if i > 0 and abs(logprob[-1] - logprob[-2]) < self.thresh:
                logger.debug("Coverged.  Logprobhistory: %s" % str(logprob))
                break
            if i == self.n_iter - 1:
                logger.debug("Finished without converging."
                             "  Logprobhistory: %s" % str(logprob))
                break

            # Maximization step
            self._do_mstep(stats, self.params)

        return self

    def _get_algorithm(self):
        "decoder algorithm"
        return self._algorithm

    def _set_algorithm(self, algorithm):
        if algorithm not in decoder_algorithms:
            raise ValueError("algorithm must be one of the decoder_algorithms")
        self._algorithm = algorithm

    algorithm = property(_get_algorithm, _set_algorithm)

    def _get_startprob(self):
        """Mixing startprob for each state."""
        return np.exp(self._log_startprob)

    def _set_startprob(self, startprob):
        if startprob is None:
            startprob = np.tile(1.0 / self.n_components, self.n_components)
        else:
            startprob = np.asarray(startprob, dtype=np.float)

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(startprob):
            normalize(startprob)

        if len(startprob) != self.n_components:
            raise ValueError('startprob must have length n_components')
        if not np.allclose(np.sum(startprob), 1.0):
            raise ValueError('startprob must sum to 1.0')

        self._log_startprob = np.log(np.asarray(startprob).copy())

    startprob_ = property(_get_startprob, _set_startprob)

    def _get_transmat(self):
        """Matrix of transition probabilities."""
        return np.exp(self._log_transmat)

    def _set_transmat(self, transmat):
        if transmat is None:
            transmat = np.tile(1.0 / self.n_components,
                               (self.n_components, self.n_components))

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(transmat):
            normalize(transmat, axis=1)

        if (np.asarray(transmat).shape
                != (self.n_components, self.n_components)):
            raise ValueError('transmat must have shape '
                             '(n_components, n_components)')
        if not np.all(np.allclose(np.sum(transmat, axis=1), 1.0)):
            raise ValueError('Rows of transmat must sum to 1.0')

        self._log_transmat = np.log(np.asarray(transmat).copy())
        underflow_idx = np.isnan(self._log_transmat)
        self._log_transmat[underflow_idx] = NEGINF

    transmat_ = property(_get_transmat, _set_transmat)

    def _do_viterbi_pass(self, framelogprob, obs = None):
        n_observations, n_components = framelogprob.shape
        state_sequence, logprob = _basehmm._viterbi(
            n_observations, n_components, self._log_startprob,
            self._log_transmat, framelogprob)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob, obs = None):
        n_observations, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_observations, n_components))
        _basehmm._forward(n_observations, n_components, self._log_startprob,
                       self._log_transmat, framelogprob, fwdlattice)
        fwdlattice[fwdlattice <= ZEROLOGPROB] = NEGINF
        return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob, obs = None):
        n_observations, n_components = framelogprob.shape
        bwdlattice = np.zeros((n_observations, n_components))
        _basehmm._backward(n_observations, n_components, self._log_startprob,
                        self._log_transmat, framelogprob, bwdlattice)

        bwdlattice[bwdlattice <= ZEROLOGPROB] = NEGINF
        
        return bwdlattice

    def _compute_log_likelihood(self, obs):
        pass

    def _generate_sample_from_state(self, state, random_state=None):
        pass

    def _init(self, obs, params):
        if 's' in params:
            self.startprob_.fill(1.0 / self.n_components)
        if 't' in params:
            self.transmat_.fill(1.0 / self.n_components)

    # Methods used by self.fit()

    def _initialize_sufficient_statistics(self):
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_components),
                 'trans': np.zeros((self.n_components, self.n_components))}
        return stats

    def _accumulate_sufficient_statistics(self, stats, seq, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        stats['nobs'] += 1
        if 's' in params:
            stats['start'] += posteriors[0]
        if 't' in params:
            n_observations, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_observations > 1:
                lneta = np.zeros((n_observations - 1, n_components, n_components))
                lnP = logsumexp(fwdlattice[-1])
                _basehmm._compute_lneta(n_observations, n_components, fwdlattice,
                                     self._log_transmat, bwdlattice, framelogprob,
                                     lnP, lneta)
                stats["trans"] += np.exp(logsumexp(lneta, 0))

    def _do_mstep(self, stats, params):
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        if self.startprob_prior is None:
            self.startprob_prior = 1.0
        if self.transmat_prior is None:
            self.transmat_prior = 1.0

        if 's' in params:
            self.startprob_ = normalize(
                np.maximum(self.startprob_prior - 1.0 + stats['start'], 1e-20))
        if 't' in params:
            transmat_ = normalize(
                np.maximum(self.transmat_prior - 1.0 + stats['trans'], 1e-20),
                axis=1)
            self.transmat_ = transmat_

            
class MultinomialHMM(BaseHMM):
    """Hidden Markov Model with multinomial (discrete) emissions

    .. warning::

       The HMM module and its functions will be removed in 0.17
       as it no longer falls within the project's scope and API.

    Attributes
    ----------
    n_components : int
        Number of states in the model.

    n_symbols : int
        Number of possible symbols emitted by the model (in the observations).

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    emissionprob : array, shape ('n_components`, 'n_symbols`)
        Probability of emitting a given symbol when in each state.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    Examples
    --------
    >>> from sklearn.hmm import MultinomialHMM
    >>> MultinomialHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    MultinomialHMM(algorithm='viterbi',...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_components=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, thresh=1e-2, params=string.ascii_letters,
                 init_params=string.ascii_letters):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_components : int
            Number of states.
        """
        BaseHMM.__init__(self, n_components, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params)

    def _get_emissionprob(self):
        """Emission probability distribution for each state."""
        return np.exp(self._log_emissionprob)

    def _set_emissionprob(self, emissionprob):
        emissionprob = np.asarray(emissionprob)
        if hasattr(self, 'n_symbols') and \
                emissionprob.shape != (self.n_components, self.n_symbols):
            raise ValueError('emissionprob must have shape '
                             '(n_components, n_symbols)')

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(emissionprob):
            normalize(emissionprob)

        self._log_emissionprob = np.log(emissionprob)
        underflow_idx = np.isnan(self._log_emissionprob)
        self._log_emissionprob[underflow_idx] = NEGINF
        self.n_symbols = self._log_emissionprob.shape[1]

    emissionprob_ = property(_get_emissionprob, _set_emissionprob)

    def _compute_log_likelihood(self, obs):
        return self._log_emissionprob[:, obs].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        rand = random_state.rand()
        symbol = (cdf > rand).argmax()
        return symbol

    def _init(self, obs, params='ste'):
        super(MultinomialHMM, self)._init(obs, params=params)
        self.random_state = check_random_state(self.random_state)

        if 'e' in params:
            if not hasattr(self, 'n_symbols'):
                symbols = set()
                for o in obs:
                    symbols = symbols.union(set(o))
                self.n_symbols = len(symbols)
            emissionprob = normalize(self.random_state.rand(self.n_components,
                                                            self.n_symbols), 1)
            self.emissionprob_ = emissionprob

    def _initialize_sufficient_statistics(self):
        stats = super(MultinomialHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_symbols))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(MultinomialHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)
        if 'e' in params:
            for t, symbol in enumerate(obs):
                stats['obs'][:, symbol] += posteriors[t]

    def _do_mstep(self, stats, params):
        super(MultinomialHMM, self)._do_mstep(stats, params)
        if 'e' in params:
            self.emissionprob_ = (stats['obs']
                                  / stats['obs'].sum(1)[:, np.newaxis])

    def _check_input_symbols(self, obs):
        """check if input can be used for Multinomial.fit input must be both
        positive integer array and every element must be continuous.
        e.g. x = [0, 0, 2, 1, 3, 1, 1] is OK and y = [0, 0, 3, 5, 10] not
        """

        symbols = np.asarray(obs).flatten()

        if symbols.dtype.kind != 'i':
            # input symbols must be integer
            return False

        if len(symbols) == 1:
            # input too short
            return False

        if np.any(symbols < 0):
            # input contains negative intiger
            return False

        symbols.sort()
        if np.any(np.diff(symbols) > 1):
            # input is discontinous
            return False

        return True

    def fit(self, obs, **kwargs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation.
        """
        err_msg = ("Input must be both positive integer array and "
                   "every element must be continuous, but %s was given.")

        if not self._check_input_symbols(obs):
            raise ValueError(err_msg % obs)

        return BaseHMM.fit(self, obs, **kwargs)


