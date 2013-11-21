from libc.math cimport exp, log
import numpy as np
cimport numpy as np
cimport cython
from .track import TrackTable
np.import_array()

ctypedef np.float64_t dtype_t
ctypedef np.int32_t itype_t

cdef dtype_t _NINF = -np.inf

def canFast(obs):
    return isinstance(obs, TrackTable) or (
        isinstance(obs, np.ndarray) and (obs.dtype == np.int32 or
                                         obs.dtype == np.uint16 or
                                         obs.dtype == np.uint8))
        
@cython.boundscheck(False)
def fastAllLogProbs(obs, logProbs, outProbs):
    if isinstance(obs, TrackTable):
        obs = obs.getNumPyArray()
    assert isinstance(obs, np.ndarray)
    assert isinstance(logProbs, np.ndarray)
    assert isinstance(outProbs, np.ndarray)
    assert len(obs.shape) == 2
    assert len(logProbs.shape) == 3
    assert logProbs.dtype == np.float
    assert outProbs.dtype == np.float
    assert outProbs.shape[0] == obs.shape[0]
    assert logProbs.shape[0] == obs.shape[1]
    cdef itype_t nObs = obs.shape[0]
    cdef itype_t nTracks = obs.shape[1]
    cdef itype_t nStates = logProbs.shape[1]

    if obs.dtype == np.int32:
        _fastAllLogProbs32(nObs, nTracks, nStates, obs, logProbs, outProbs)
    elif obs.dtype == np.uint16:
        _fastAllLogProbsU16(nObs, nTracks, nStates, obs, logProbs, outProbs)
    elif obs.dtype == np.uint8:
        _fastAllLogProbsU8(nObs, nTracks, nStates, obs, logProbs, outProbs)
    else:
        print obs.dtype
        assert False

@cython.boundscheck(False)
def _fastAllLogProbsU8(itype_t nObs, itype_t nTracks, itype_t nStates,
                      np.ndarray[np.uint8_t, ndim=2] obs,
                      np.ndarray[dtype_t, ndim=3] logProbs,
                      np.ndarray[dtype_t, ndim=2] outProbs):
    for i in xrange(nObs):
       for j in xrange(nStates):
           outProbs[i,j] = 0.0
           for k in xrange(nTracks):
               outProbs[i, j] += logProbs[k, j, obs[i, k]]
       
@cython.boundscheck(False)
def _fastAllLogProbsU16(itype_t nObs, itype_t nTracks, itype_t nStates,
                      np.ndarray[np.uint16_t, ndim=2] obs,
                      np.ndarray[dtype_t, ndim=3] logProbs,
                      np.ndarray[dtype_t, ndim=2] outProbs):
    for i in xrange(nObs):
       for j in xrange(nStates):
           outProbs[i,j] = 0.0
           for k in xrange(nTracks):
               outProbs[i, j] += logProbs[k, j, obs[i, k]]

@cython.boundscheck(False)
def _fastAllLogProbs32(itype_t nObs, itype_t nTracks, itype_t nStates,
                      np.ndarray[np.int32_t, ndim=2] obs,
                      np.ndarray[dtype_t, ndim=3] logProbs,
                      np.ndarray[dtype_t, ndim=2] outProbs):
    for i in xrange(nObs):
       for j in xrange(nStates):
           outProbs[i,j] = 0.0
           for k in xrange(nTracks):
               outProbs[i, j] += logProbs[k, j, obs[i, k]]

@cython.boundscheck(False)
def fastAccumulateStats(obs, obsStats, posteriors):
    if isinstance(obs, TrackTable):
        obs = obs.getNumPyArray()
    assert isinstance(obs, np.ndarray)

    assert len(obs.shape) == 2

    cdef itype_t nObs = obs.shape[0]
    cdef itype_t nTracks = obs.shape[1]
    cdef itype_t nStates = obsStats[0].shape[0]

    if obs.dtype == np.int32:
        _fastAccumulateStats32(nObs, nTracks, nStates, obs, obsStats,
                               posteriors)
    elif obs.dtype == np.uint16:
        _fastAccumulateStatsU16(nObs, nTracks, nStates, obs, obsStats,
                               posteriors)
    elif obs.dtype == np.uint8:
        _fastAccumulateStatsU8(nObs, nTracks, nStates, obs, obsStats,
                               posteriors)
    else:
        assert False

@cython.boundscheck(False)
def _fastAccumulateStatsU8(itype_t nObs, itype_t nTracks, itype_t nStates,
                           np.ndarray[np.uint8_t, ndim=2] obs, obsStats,
                           posteriors):
    for i in xrange(nObs):
        for track in xrange(nTracks):
            for state in xrange(nStates):
                obsVal = obs[i,track]
                obsStats[track][state, obsVal] += posteriors[i, state]

@cython.boundscheck(False)
def _fastAccumulateStatsU16(itype_t nObs, itype_t nTracks, itype_t nStates,
                           np.ndarray[np.uint16_t, ndim=2] obs, obsStats,
                           posteriors):
    for i in xrange(nObs):
        for track in xrange(nTracks):
            for state in xrange(nStates):
                obsVal = obs[i,track]
                obsStats[track][state, obsVal] += posteriors[i, state]

@cython.boundscheck(False)
def _fastAccumulateStats32(itype_t nObs, itype_t nTracks, itype_t nStates,
                            np.ndarray[np.int32_t, ndim=2] obs, obsStats,
                            posteriors):
    for i in xrange(nObs):
        for track in xrange(nTracks):
            for state in xrange(nStates):
                obsVal = obs[i,track]
                obsStats[track][state, obsVal] += posteriors[i, state]
