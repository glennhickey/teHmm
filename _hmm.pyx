#Released under the MIT license, see LICENSE.txt
#Copyright (C) 2014 by Glenn Hickey

"""
Reimplement the Cython HMM dynamic programming routines from Scikit-learn
(which are preserved in _basehmm.pyx) to be faster.  The changes made have
a 10x - 100x speed increase on my development machine (Mavericks Macbook air).
They are centered around absolutely avoiding function calls in the inner loops,
including any numpy vector operations.  Traceback pointers were added to Viterbi
to further speed up the algorithm (albeit at the cost of O(N) memory).

Improvements can be measured using tests/dpBenchmark.py

-- Glenn Hickey, 2014

 Derived from scikit-learn/sklearn/_hmmc.pyx
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
"""

from libc.math cimport exp, log
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

np.import_array()

ctypedef np.float64_t dtype_t

cdef dtype_t _NINF = -np.inf
cdef dtype_t ZEROLOGPROB = -1e200

@cython.boundscheck(False)
def _log_sum_lneta(int n_observations, int n_components,
        np.ndarray[dtype_t, ndim=2] fwdlattice,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] bwdlattice,
        np.ndarray[dtype_t, ndim=2] framelogprob,
        double logprob,
        np.ndarray[dtype_t, ndim=1] segRatios,
        np.ndarray[dtype_t, ndim=2] logsum_lneta):
	# This is a combined version of _compute_lneta (original preserved in
	# _basehmm.pyx) and logsumexp (original preserved in basehmm.py).  We 
	# rewrite to use two passes to avoid the giant memory overhead (
	# original required O(obs * states * states)).  
    cdef int i, j, t, hasRatios = 0
    cdef np.ndarray[dtype_t, ndim = 2] maxMatrix
    cdef double x
    maxMatrix = _NINF + np.zeros((n_components, n_components))
    if segRatios is not None:
        hasRatios = 1
    
    # find max
    for t in range(n_observations - 1):
        for i in range(n_components):
            for j in range(n_components):
                x = fwdlattice[t, i] + log_transmat[i, j] \
                  + framelogprob[t + 1, j] + bwdlattice[t + 1, j] - logprob
                if hasRatios == 1:
                    x += log_transmat[j, j] * segRatios[t + 1]
                    if i == j:
                        x += segRatios[t + 1]
                if x > maxMatrix[i, j]:
                    maxMatrix[i, j] = x

    # sum exp(x - max)
    for t in xrange(0, n_observations - 1):
        for i in xrange(0, n_components):
            for j in xrange(0, n_components):
                x = fwdlattice[t, i] + log_transmat[i, j] \
                    + framelogprob[t + 1, j] + bwdlattice[t + 1, j] - logprob
                if hasRatios == 1:
                    x += log_transmat[j, j] * segRatios[t + 1]
                    if i == j:
                        x += segRatios[t + 1]
                logsum_lneta[i, j] += exp(x - maxMatrix[i, j])

    # return log(sum(x-max)) + max
    for i in xrange(0, n_components):
        for j in xrange(0, n_components):
            logsum_lneta[i, j] = log(logsum_lneta[i, j]) + maxMatrix[i, j]

            
@cython.boundscheck(False)
def _forward(int n_observations, int n_components,
        np.ndarray[dtype_t, ndim=1] log_startprob,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] framelogprob,
        np.ndarray[dtype_t, ndim=1] segRatios,
        np.ndarray[dtype_t, ndim=2] fwdlattice):

    cdef int t, i, j, hasRatios = 0
    cdef double logprob
    cdef dtype_t vmax = 0
    cdef dtype_t power_sum = 0.0
    cdef double* work_buffer = <double *> \
      malloc(n_components * cython.sizeof(double))
    if segRatios is not None:
        hasRatios = 1

    for i in xrange(n_components):
        fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]
        if hasRatios == 1:
            fwdlattice[0, i] += log_transmat[i, i] * segRatios[0]

    for t in xrange(1, n_observations):
        for j in xrange(n_components):
            vmax = _NINF
            for i in xrange(n_components):
                work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j]
                if work_buffer[i] > vmax:
                    vmax = work_buffer[i]
            power_sum = 0.0
            for i in xrange(n_components):
                power_sum += exp(work_buffer[i] - vmax)                
            fwdlattice[t, j] = log(power_sum) + vmax + framelogprob[t, j]
            if hasRatios == 1:
                fwdlattice[t, j] += log_transmat[j, j] * segRatios[j]
            if fwdlattice[t, j] <= ZEROLOGPROB:
                fwdlattice[t, j] = _NINF
    free(work_buffer)

@cython.boundscheck(False)
def _backward(int n_observations, int n_components,
        np.ndarray[dtype_t, ndim=1] log_startprob,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] framelogprob,
        np.ndarray[dtype_t, ndim=1] segRatios,
        np.ndarray[dtype_t, ndim=2] bwdlattice):

    cdef int t, i, j, hasRatios = 0
    cdef double logprob
    cdef dtype_t vmax = 0
    cdef dtype_t power_sum = 0.0
    cdef double* work_buffer = <double *> \
      malloc(n_components * cython.sizeof(double))
    if segRatios is not None:
        hasRatios = 1

    for i in xrange(n_components):
        bwdlattice[n_observations - 1, i] = log(1. / float(n_components))

    for t in xrange(n_observations - 2, -1, -1):
        for i in xrange(n_components):
            vmax = _NINF
            for j in xrange(n_components):
                work_buffer[j] = log_transmat[i, j] + framelogprob[t + 1, j] \
                    + bwdlattice[t + 1, j]
                if hasRatios == 1:
                    work_buffer[j] += log_transmat[j, j] * segRatios[t + 1]
                if work_buffer[j] > vmax:
                    vmax = work_buffer[j]
            power_sum = 0.0
            for j in xrange(n_components):
                power_sum += exp(work_buffer[j] - vmax)
            bwdlattice[t, i] = log(power_sum) + vmax
            if bwdlattice[t, i] <= ZEROLOGPROB:
                bwdlattice[t, i] = _NINF

    free(work_buffer)


@cython.boundscheck(False)
def _viterbi(int n_observations, int n_components,
        np.ndarray[dtype_t, ndim=1] log_startprob,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=1] segRatios,
        np.ndarray[dtype_t, ndim=2] framelogprob):

    cdef int t, max_pos, hasRatios = 0
    cdef np.ndarray[dtype_t, ndim = 2] viterbi_lattice
    cdef np.ndarray[np.int_t, ndim = 1] state_sequence
    cdef np.ndarray[np.int16_t, ndim = 2] trace_back
    cdef dtype_t logprob
    cdef dtype_t maxprob
    cdef dtype_t curprob
    cdef np.int16_t maxState
    if segRatios is not None:
        hasRatios = 1

    # Initialization
    state_sequence = np.empty(n_observations, dtype=np.int)
    viterbi_lattice = np.zeros((n_observations, n_components))
    viterbi_lattice[0] = log_startprob + framelogprob[0]
    if hasRatios == 1:
        for toState in xrange(0, n_components):
            viterbi_lattice[toState] += log_transmat[toState, toState] * segRatios[0]

    trace_back = np.empty((n_observations, n_components), dtype=np.int16)

    # Induction
    for t in xrange(1, n_observations):
        for toState in xrange(0, n_components):            
            maxprob = viterbi_lattice[t-1, 0] + log_transmat[0, toState] +\
              framelogprob[t, toState]
            if hasRatios == 1:
                maxprob += log_transmat[toState, toState] * segRatios[t]
            maxState = 0
            for fromState in xrange(1, n_components):
                curprob = viterbi_lattice[t-1, fromState] + \
                  log_transmat[fromState, toState] +\
                  framelogprob[t, toState]
                if hasRatios == 1:
                    curprob += log_transmat[toState, toState] * segRatios[t]
                if curprob > maxprob:
                    maxprob = curprob
                    maxState = fromState
            viterbi_lattice[t, toState] = maxprob
            trace_back[t, toState] = maxState
            
    # Observation traceback
    max_pos = np.argmax(viterbi_lattice[n_observations - 1, :])
    state_sequence[n_observations - 1] = max_pos
    logprob = viterbi_lattice[n_observations - 1, max_pos]

    for t in xrange(n_observations - 1, 0, -1):
        state_sequence[t - 1] = trace_back[t, state_sequence[t]]

    return state_sequence, logprob
